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
    last_token: list[float]
    layer: int
    hidden_dim: int
    model: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    hidden_state: Optional[HiddenStateInfo] = None


# --- Injection hook ---

def make_injection_hook(v_hat_tensor: torch.Tensor, alpha: float, mode: str = "all"):
    """创建 forward hook，在指定层的输出上注入 alpha * v_hat。

    Args:
        v_hat_tensor: 归一化方向向量 (hidden_dim,)，已在 GPU 上
        alpha: 注入强度
        mode: "all" = 每个 generation step 都注入；
              "prefill_only" = 仅在 prefill 阶段（seq_len > 1）注入
    """
    def hook(module, input, output):
        hidden_states = output[0]  # (batch, seq_len, hidden_dim)
        if mode == "prefill_only" and hidden_states.shape[1] == 1:
            return output  # autoregressive step, skip
        # 修改 last token position 的 hidden state
        hidden_states[:, -1, :] += alpha * v_hat_tensor
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
            target_layer = model.model.layers[request.injection_layer]
            hook_fn = make_injection_hook(v_hat, request.injection_alpha, request.injection_mode)
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

            # Extract last-token hidden state from last layer.
            # outputs.hidden_states is a tuple of (num_generated_tokens) elements.
            # Each element is a tuple of (num_layers+1) tensors.
            # For each generation step after the first, tensor shape is (batch, 1, hidden_dim).
            last_step_hs = outputs.hidden_states[-1]  # last generation step
            last_layer_hs = last_step_hs[-1]           # last layer
            last_token_hs = last_layer_hs[:, -1, :]    # (batch, hidden_dim)
            hidden_dim = last_token_hs.shape[-1]

            # Convert to CPU list BEFORE freeing GPU memory
            hs_list = last_token_hs[0].float().cpu().tolist()

            completion_tokens = generated_ids.shape[1]

            # Free GPU memory: hidden_states from all generation steps are huge (~1.5GB+)
            del outputs, last_step_hs, last_layer_hs, last_token_hs
            del inputs, generated_ids
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        injection_info = ""
        if injection_active:
            injection_info = (f", injection=[id={request.injection_id}, "
                              f"alpha={request.injection_alpha}, mode={request.injection_mode}, "
                              f"layer={request.injection_layer}]")
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
    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.model_path, device_map=args.device_map, load_in_4bit=args.load_in_4bit)

    app_state["model"] = model
    app_state["processor"] = processor
    app_state["model_name"] = args.model_path.rstrip("/").split("/")[-1]

    hidden_dim = getattr(model.config, "hidden_size", None)
    logger.info(f"Model: {app_state['model_name']}, hidden_dim={hidden_dim}")
    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
