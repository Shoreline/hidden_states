"""
Qwen2.5-VL model loader with hidden states support.
"""
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

MODEL_PATH = "/home/ubuntu/models/qwen2.5-vl-fp8"


def _convert_fp8_to_bf16(model):
    """
    Convert any residual float8 parameters/buffers to bfloat16.
    This fixes dtype mismatch issues with FP8 models.
    """
    for p in model.parameters():
        if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            p.data = p.data.to(torch.bfloat16)
    
    for b in model.buffers():
        if b.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            b.data = b.data.to(torch.bfloat16)
    
    return model


def load_model(model_path: str = MODEL_PATH, device_map: str = "cuda"):
    """
    Load Qwen2.5-VL model with hidden states output enabled.
    
    Args:
        model_path: Path to the model directory
        device_map: Device mapping strategy ("cuda", "auto", "cuda:0", etc.)
    
    Returns:
        tuple: (model, processor)
    """
    print(f"Loading model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    
    # Convert residual FP8 parameters to bfloat16
    model = _convert_fp8_to_bf16(model)
    
    print(f"Model loaded successfully")
    return model, processor


def get_hidden_states(model, processor, text: str, image=None):
    """
    Extract hidden states from all layers for given input.
    
    Args:
        model: The loaded Qwen model
        processor: The processor for the model
        text: Input text
        image: Optional input image (PIL Image or path)
    
    Returns:
        tuple: Hidden states from all layers
               Shape: (num_layers+1, batch, seq_len, hidden_dim)
    """
    # Prepare inputs
    if image is not None:
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    return outputs.hidden_states


def get_last_token_hidden_state(model, processor, text: str, image=None, layer: int = -1):
    """
    Extract hidden state of the last token from a specific layer.
    
    Args:
        model: The loaded Qwen model
        processor: The processor for the model
        text: Input text
        image: Optional input image (PIL Image or path)
        layer: Layer index to extract from (-1 for last layer)
    
    Returns:
        torch.Tensor: Hidden state of last token, shape (batch, hidden_dim)
    """
    hidden_states = get_hidden_states(model, processor, text, image)
    # hidden_states[layer] has shape (batch, seq_len, hidden_dim)
    # We take [:, -1, :] to get the last token's hidden state
    return hidden_states[layer][:, -1, :]


def get_attention_weights(model, processor, text: str, image=None):
    """
    Extract attention weights from all layers for given input.
    
    Args:
        model: The loaded Qwen model
        processor: The processor for the model
        text: Input text
        image: Optional input image
    
    Returns:
        tuple: Attention weights from all layers
    """
    if image is not None:
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    return outputs.attentions


if __name__ == "__main__":
    # Quick test
    model, processor = load_model()
    
    prompt = "Say hello in one word"
    hidden_states = get_hidden_states(model, processor, prompt)
    print(f"Number of layers: {len(hidden_states)}")
    print(f"Hidden state shape: {hidden_states[0].shape}")
    
    # Get last token hidden state from last layer
    last_hidden = get_last_token_hidden_state(model, processor, prompt)
    print(f"\nLast token hidden state:")
    print(f"  Shape: {last_hidden.shape}")
    print(f"  Dtype: {last_hidden.dtype}")
    print(f"  Norm: {torch.norm(last_hidden).item():.4f}")
