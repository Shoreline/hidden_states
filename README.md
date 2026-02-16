# Qwen Hidden States Analysis

分析和可视化 VL 模型的 hidden states。支持任意 HuggingFace VL 模型。

## Setup

```bash
pip install -r requirements.txt
```

## Models

| Path | Description |
|------|-------------|
| `/home/ubuntu/models/Qwen2.5-VL-7B-Instruct` | BF16 unquantized, use `--device_map auto` on L4 23GB |
| `/home/ubuntu/models/qwen2.5-vl-fp8` | FP8 (modelopt), broken with transformers 5.0.0 |

## API Server

Serves OpenAI-compatible `/v1/chat/completions` with hidden state extraction.

### Start the server (on AWS)

```bash
cd /home/ubuntu/projects/hidden_states
source venv/bin/activate
python server.py --model_path /home/ubuntu/models/Qwen2.5-VL-7B-Instruct --device_map auto

python server.py --model_path /root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 --device_map auto

```

### Test from local Mac

Port 8000 is open to your IP in the AWS security group.

```bash
# Health check
curl http://34.210.214.193:8000/health

# Chat completion with hidden states
curl -s http://34.210.214.193:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32,
    "temperature": 0.7
  }' | python3 -m json.tool
```

Response includes standard OpenAI fields plus `hidden_state`:
```json
{
    "hidden_state": {
        "last_token": [0.621, 0.301, ...],
        "layer": -1,
        "hidden_dim": 3584,
        "model": "Qwen2.5-VL-7B-Instruct"
    }
}
```

### Test directly on AWS

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32,
    "temperature": 0.7
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
print('Response:', r['choices'][0]['message']['content'])
print('Hidden dim:', r['hidden_state']['hidden_dim'])
print('Vector len:', len(r['hidden_state']['last_token']))
"
```

## Python API (direct model access)

```python
from src.model_loader import load_model, get_hidden_states, get_last_token_hidden_state

model, processor = load_model()
hidden_states = get_hidden_states(model, processor, "Hello world")
last_hs = get_last_token_hidden_state(model, processor, "Hello world")
```
