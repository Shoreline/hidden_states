# Qwen Hidden States Analysis

分析和可视化 Qwen2.5-VL 模型的 hidden states。

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.model_loader import load_model, get_hidden_states

model, processor = load_model()
hidden_states = get_hidden_states(model, processor, "Hello world")
```

## Model Path

Model location: `/home/ubuntu/models/qwen2.5-vl-fp8`
