"""
Model-agnostic FastAPI server for VL models.
Serves OpenAI-compatible /v1/chat/completions with hidden state extraction.
Supports activation injection for causal verification experiments.

Usage:
    python server.py --model_path /home/ubuntu/models/qwen2.5-vl-fp8
    python server.py --model_path /path/to/qwen3-vl-235b --device_map auto
"""
import argparse
import base64
import io
import logging
import os
import threading
import time
import uuid

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Union

from src.model_loader import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Global state ---
app_state = {
    "model": None,
    "processor": None,
    "model_name": None,
}

app = FastAPI(title="Hidden States VL Server")

# GPU 推理锁：单卡只能串行推理，用锁保护避免多线程同时 generate 导致 OOM
_inference_lock = threading.Lock()

# 全局开关：强制所有请求返回全层 hidden states（通过 --output_all_layers 或 env 控制）
_force_all_layers = False

# 注入向量存储：{injection_id: torch.Tensor on GPU}
_injection_vectors: dict[str, torch.Tensor] = {}
_injection_counter = 0


# --- Request/Response models ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, list]


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    # Injection parameters (optional)
    injection_id: Optional[str] = None
    injection_alpha: float = 0.0
    injection_mode: str = "all"       # "all" or "prefill_only"
    injection_layer: int = -1         # target layer index
    injection_relative: bool = True   # True: alpha 相对于 ||h||; False: 绝对值
    output_all_layers: bool = False   # True: 返回所有层 hidden states（base64 numpy）


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class HiddenStateInfo(BaseModel):
    last_token: list[float]              # 单层 hidden state（向后兼容）
    layer: int                           # 来源层（-1 = 最后一层）
    hidden_dim: int
    model: str
    all_layers_b64: Optional[str] = None # 所有层 hidden states，base64 编码的 float32 numpy (num_layers, hidden_dim)
    num_layers: Optional[int] = None     # decoder layer 数量（不含 embedding）


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    hidden_state: Optional[HiddenStateInfo] = None


# --- Injection hook ---

def make_injection_hook(v_hat_tensor: torch.Tensor, alpha: float, mode: str = "all",
                        relative: bool = True):
    """创建 forward hook，在指定层的输出上注入方向向量。

    Args:
        v_hat_tensor: 归一化方向向量 (hidden_dim,)，已在 GPU 上
        alpha: 注入强度
        mode: "all" = 每个 generation step 都注入；
              "prefill_only" = 仅在 prefill 阶段（seq_len > 1）注入
        relative: True = alpha 相对于 ||h||（如 0.05 = 5% of hidden state norm）；
                  False = alpha 为绝对值

    注意：不同模型的 decoder layer forward 返回格式不同：
    - 有些返回 tuple: (hidden_states, ...)，hidden_states 为 3D (batch, seq, dim)
    - Qwen3-VL 直接返回 tensor: hidden_states 为 2D (seq, dim) 或 3D
    Hook 需要兼容各种情况。
    """
    _call_count = [0]

    def _get_last_token(hidden_states):
        """获取 last token 的 hidden state 引用（支持 2D 和 3D）。"""
        if hidden_states.ndim == 3:
            return hidden_states[:, -1, :]  # (batch, dim)
        else:
            return hidden_states[-1:, :]    # (1, dim)

    def _should_skip(hidden_states):
        """prefill_only 模式：autoregressive step 时跳过。"""
        if mode != "prefill_only":
            return False
        seq_dim = 1 if hidden_states.ndim == 3 else 0
        return hidden_states.shape[seq_dim] == 1

    def _inject(hidden_states):
        """执行注入并记录诊断。"""
        _call_count[0] += 1
        last_h = _get_last_token(hidden_states)
        h_norm = last_h.norm().item()

        if relative:
            delta = (alpha * h_norm) * v_hat_tensor
        else:
            delta = alpha * v_hat_tensor

        last_h += delta

        if _call_count[0] <= 3:
            delta_norm = delta.norm().item()
            ratio = delta_norm / h_norm * 100 if h_norm > 0 else 0
            logger.info(f"[injection #{_call_count[0]}] shape={tuple(hidden_states.shape)}, "
                        f"||h||={h_norm:.1f}, ||delta||={delta_norm:.1f} ({ratio:.1f}%), "
                        f"relative={relative}")

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if _should_skip(output):
                return output
            _inject(output)
            return output
        else:
            hidden_states = output[0]
            if _should_skip(hidden_states):
                return output
            _inject(hidden_states)
            return (hidden_states,) + output[1:]

    return hook


# --- Helpers ---

def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a base64 data URL, raw base64 string, or HTTP URL to PIL Image."""
    if data_url.startswith("http://") or data_url.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(data_url) as resp:
            image_bytes = resp.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if data_url.startswith("data:"):
        # data:image/jpeg;base64,/9j/4AAQ...
        _, b64_data = data_url.split(",", 1)
    else:
        b64_data = data_url
    image_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def parse_openai_messages(messages: list[ChatMessage]):
    """
    Convert OpenAI-format messages to Qwen processor format.

    Returns:
        (qwen_messages, images): messages for apply_chat_template, and
        a list of PIL images in the order they appear.
    """
    qwen_messages = []
    images = []

    for msg in messages:
        if isinstance(msg.content, str):
            qwen_messages.append({
                "role": msg.role,
                "content": [{"type": "text", "text": msg.content}],
            })
        elif isinstance(msg.content, list):
            content_parts = []
            for part in msg.content:
                if not isinstance(part, dict):
                    content_parts.append({"type": "text", "text": str(part)})
                    continue

                ptype = part.get("type", "")
                if ptype == "text":
                    content_parts.append({"type": "text", "text": part["text"]})
                elif ptype == "image_url":
                    image_url = part["image_url"]
                    url = image_url if isinstance(image_url, str) else image_url.get("url", "")
                    img = decode_base64_image(url)
                    images.append(img)
                    # Placeholder for chat template — it inserts vision tokens
                    content_parts.append({"type": "image"})
                elif ptype == "image":
                    # Already in Qwen format — could be a URL or path
                    content_parts.append(part)

            qwen_messages.append({
                "role": msg.role,
                "content": content_parts,
            })

    return qwen_messages, images


# --- Admin endpoints: injection vector management ---

@app.post("/admin/load_injection_vector")
async def load_injection_vector(vector: UploadFile = File(...)):
    """上传 .npy 向量文件，加载到 GPU 内存。返回 injection_id。"""
    global _injection_counter
    try:
        content = await vector.read()
        arr = np.load(io.BytesIO(content))
        if arr.ndim != 1:
            return JSONResponse(status_code=400, content={"error": f"Expected 1D array, got shape {arr.shape}"})

        model = app_state["model"]
        device = next(model.parameters()).device
        tensor = torch.from_numpy(arr.astype(np.float32)).to(device)

        _injection_counter += 1
        injection_id = f"v{_injection_counter}"
        _injection_vectors[injection_id] = tensor

        logger.info(f"Loaded injection vector: id={injection_id}, dim={arr.shape[0]}, "
                     f"norm={np.linalg.norm(arr):.4f}, device={device}")
        return {
            "injection_id": injection_id,
            "hidden_dim": arr.shape[0],
            "norm": float(np.linalg.norm(arr)),
        }
    except Exception as e:
        logger.exception("Failed to load injection vector")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/admin/injection_vectors")
async def list_injection_vectors():
    """列出已加载的注入向量。"""
    vectors = {}
    for vid, tensor in _injection_vectors.items():
        vectors[vid] = {
            "hidden_dim": tensor.shape[0],
            "device": str(tensor.device),
        }
    return {"vectors": vectors}


@app.post("/admin/set_output_all_layers")
async def set_output_all_layers(enabled: bool = True):
    """动态开启/关闭全层 hidden states 输出。"""
    global _force_all_layers
    _force_all_layers = enabled
    logger.info(f"output_all_layers set to {enabled}")
    return {"output_all_layers": enabled}


@app.get("/admin/config")
async def get_admin_config():
    """查看当前服务端配置。"""
    return {
        "output_all_layers": _force_all_layers,
        "injection_vectors": list(_injection_vectors.keys()),
        "model": app_state.get("model_name"),
        "decoder_layers_path": app_state.get("decoder_layers_path"),
    }


@app.delete("/admin/injection_vectors/{injection_id}")
async def delete_injection_vector(injection_id: str):
    """删除已加载的注入向量。"""
    if injection_id not in _injection_vectors:
        return JSONResponse(status_code=404, content={"error": f"Unknown injection_id: {injection_id}"})
    del _injection_vectors[injection_id]
    logger.info(f"Deleted injection vector: {injection_id}")
    return {"deleted": injection_id}


# --- Chat completion ---

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions with hidden state extraction.

    使用同步 def（非 async def），FastAPI 自动在线程池中执行，
    不阻塞 event loop，健康检查和连接管理保持响应。
    _inference_lock 保证单卡 GPU 推理串行，避免 OOM。
    """
    t0 = time.time()
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    model = app_state["model"]
    processor = app_state["processor"]

    try:
        # Parse messages（含 urllib 图片下载，可并行于其他请求的 GPU 推理）
        qwen_messages, images = parse_openai_messages(request.messages)

        # Apply chat template
        text = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize + process images
        if images:
            inputs = processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text],
                return_tensors="pt",
            )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # 准备 injection hook（如果请求了注入）
        hook_handle = None
        injection_active = False
        if request.injection_id and request.injection_alpha != 0.0:
            v_hat = _injection_vectors.get(request.injection_id)
            if v_hat is None:
                raise HTTPException(status_code=400,
                                    detail=f"Unknown injection_id: {request.injection_id}")
            decoder_layers = app_state.get("decoder_layers")
            if decoder_layers is None:
                raise HTTPException(status_code=500, detail="Decoder layers not found — injection unavailable")
            target_layer = decoder_layers[request.injection_layer]
            hook_fn = make_injection_hook(v_hat, request.injection_alpha, request.injection_mode,
                                           relative=request.injection_relative)
            hook_handle = target_layer.register_forward_hook(hook_fn)
            injection_active = True

        # GPU 推理串行：用锁保护，避免多线程同时 generate
        with _inference_lock:
            try:
                # Generate with hidden states
                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=request.max_tokens,
                        top_p=request.top_p,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )
                    if request.temperature > 0:
                        gen_kwargs["temperature"] = request.temperature
                        gen_kwargs["do_sample"] = True
                    else:
                        gen_kwargs["do_sample"] = False

                    outputs = model.generate(**inputs, **gen_kwargs)
            finally:
                # 确保 hook 一定被移除
                if hook_handle is not None:
                    hook_handle.remove()

            # Decode response text
            generated_ids = outputs.sequences[:, input_len:]
            response_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Extract hidden states.
            # outputs.hidden_states is a tuple of (num_generated_tokens) elements.
            # Each element is a tuple of (num_layers+1) tensors.
            # Layer 0 = embedding, layers 1..L = decoder layers.
            last_step_hs = outputs.hidden_states[-1]  # last generation step
            last_layer_hs = last_step_hs[-1]           # last layer
            last_token_hs = last_layer_hs[:, -1, :]    # (batch, hidden_dim)
            hidden_dim = last_token_hs.shape[-1]

            # 单层 hidden state（向后兼容）
            hs_list = last_token_hs[0].float().cpu().tolist()

            # 所有层 hidden states（可选）
            all_layers_b64 = None
            num_layers = len(last_step_hs) - 1  # 不含 embedding 层
            if request.output_all_layers or _force_all_layers:
                # 提取每层 last token: (num_layers, hidden_dim)，跳过 layer 0 (embedding)
                all_layers = torch.stack(
                    [last_step_hs[l][:, -1, :].squeeze(0) for l in range(1, len(last_step_hs))]
                ).float().cpu().numpy()  # (num_layers, hidden_dim)
                all_layers_b64 = base64.b64encode(all_layers.astype(np.float32).tobytes()).decode('ascii')
                logger.info(f"All-layer hidden states: shape={all_layers.shape}, "
                            f"b64_len={len(all_layers_b64)}")

            completion_tokens = generated_ids.shape[1]

            # Free GPU memory
            del outputs, last_step_hs, last_layer_hs, last_token_hs
            del inputs, generated_ids
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        injection_info = ""
        if injection_active:
            injection_info = (f", injection=[id={request.injection_id}, "
                              f"alpha={request.injection_alpha}, mode={request.injection_mode}, "
                              f"layer={request.injection_layer}, "
                              f"relative={request.injection_relative}]")
        logger.info(
            f"[{elapsed:.1f}s] prompt={input_len} tokens, generated={completion_tokens} tokens, "
            f"hidden_dim={hidden_dim}{injection_info}, preview={response_text[:80]!r}"
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=app_state["model_name"],
            choices=[Choice(
                message=ChoiceMessage(content=response_text),
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=input_len,
                completion_tokens=completion_tokens,
                total_tokens=input_len + completion_tokens,
            ),
            hidden_state=HiddenStateInfo(
                last_token=hs_list,
                layer=-1,
                hidden_dim=hidden_dim,
                model=app_state["model_name"],
                all_layers_b64=all_layers_b64,
                num_layers=num_layers,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during inference")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": app_state["model_name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": app_state["model_name"]}


def main():
    parser = argparse.ArgumentParser(description="Hidden States VL Server")
    parser.add_argument("--model_path", required=True, help="Path to HuggingFace VL model")
    parser.add_argument("--device_map", default="cuda", help="Device map (cuda, auto, cpu)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization (saves ~75%% VRAM)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output_all_layers", action="store_true",
                       help="强制所有请求返回全层 hidden states（也可通过 env OUTPUT_ALL_LAYERS=1 设置）")
    args = parser.parse_args()

    # 全局开关
    global _force_all_layers
    _force_all_layers = args.output_all_layers or os.environ.get("OUTPUT_ALL_LAYERS") == "1"

    # Load model
    model, processor = load_model(args.model_path, device_map=args.device_map, load_in_4bit=args.load_in_4bit)

    app_state["model"] = model
    app_state["processor"] = processor
    app_state["model_name"] = args.model_path.rstrip("/").split("/")[-1]

    # 探测 decoder layers 路径（不同 VL 模型结构不同）
    # Qwen3-VL: model.model.language_model.layers
    # Qwen2.5-VL: model.model.layers
    decoder_layers = None
    for path in ("model.language_model.layers", "model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            decoder_layers = obj
            app_state["decoder_layers_path"] = path
            break
        except AttributeError:
            continue
    if decoder_layers is None:
        logger.warning("Could not find decoder layers — injection will not work")
    else:
        logger.info(f"Decoder layers: model.{path} ({len(decoder_layers)} layers)")
    app_state["decoder_layers"] = decoder_layers

    hidden_dim = getattr(model.config, "hidden_size", None)
    logger.info(f"Model: {app_state['model_name']}, hidden_dim={hidden_dim}")
    if _force_all_layers:
        logger.info("output_all_layers: ENABLED (all requests return full-layer hidden states)")
    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
