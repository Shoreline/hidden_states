"""
Qwen2.5-VL model loader with hidden states support.
"""
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

MODEL_PATH = "/home/ubuntu/models/qwen2.5-vl-fp8"


def load_model(model_path: str = MODEL_PATH, device_map: str = "auto"):
    """
    Load Qwen2.5-VL model with hidden states output enabled.
    
    Args:
        model_path: Path to the model directory
        device_map: Device mapping strategy ("auto", "cuda:0", etc.)
    
    Returns:
        tuple: (model, processor)
    """
    print(f"Loading model from {model_path}...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"Model loaded on {model.device}")
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
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states


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
        outputs = model(**inputs, output_attentions=True)
    
    return outputs.attentions


if __name__ == "__main__":
    # Quick test
    model, processor = load_model()
    hidden_states = get_hidden_states(model, processor, "Hello, world!")
    print(f"Number of layers: {len(hidden_states)}")
    print(f"Hidden state shape: {hidden_states[0].shape}")
