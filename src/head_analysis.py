"""
Attention head activation analysis.

Two complementary metrics:
  1. Head contribution norm  — the L2 norm of what each head writes to the
     residual stream.  Large norm → head is doing meaningful work.
  2. Attention entropy       — Shannon entropy of each head's attention
     distribution.  Low entropy → head is focused on specific tokens.

Both are computed for a single forward pass (no generation needed).
Shape of returned arrays: (num_layers, num_heads)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def get_head_activations(model, inputs, token_idx: int = -1) -> dict:
    """
    Analyse attention heads for a single forward pass.

    Args:
        model:      Loaded HuggingFace causal LM / VL model.
        inputs:     Dict of tensors already on model.device
                    (output of processor(..., return_tensors='pt')).
        token_idx:  Which token position to analyse.
                    -1 = last token (default, matches what the server returns).

    Returns dict with:
        contrib_norms  : np.ndarray  (num_layers, num_heads)
                         L2 norm of each head's contribution to residual stream.
        attn_entropy   : np.ndarray  (num_layers, num_heads)
                         Shannon entropy of attention weights.
        attn_weights   : list of np.ndarray, each (num_heads, seq_q, seq_k)
                         Raw attention weight matrices per layer.
        num_layers     : int
        num_heads      : int
        seq_len        : int
    """
    # ---- collect o_proj inputs (pre-projection per-head activations) ----
    oproj_inputs = {}   # layer_idx -> tensor (batch, seq, num_heads*head_dim)

    def make_oproj_hook(layer_idx):
        def hook(module, inp, _out):
            # inp[0]: (batch, seq, num_heads * head_dim) — concatenated head outputs
            oproj_inputs[layer_idx] = inp[0].detach().float()
        return hook

    handles = []
    # Qwen3VL wraps the text decoder at model.model.language_model.layers
    lm = getattr(model.model, "language_model", model.model)
    layers = lm.layers
    for i, layer in enumerate(layers):
        h = layer.self_attn.o_proj.register_forward_hook(make_oproj_hook(i))
        handles.append(h)

    # Also hook attention weights if the model uses eager attention.
    # With sdpa (default), output_attentions is not supported — we silently skip entropy.
    attn_weight_captures = {}

    def make_attn_weight_hook(layer_idx):
        def hook(module, inp, out):
            # output[1] is attn_weights when output_attentions=True, else None
            if isinstance(out, (tuple, list)) and len(out) > 1 and out[1] is not None:
                attn_weight_captures[layer_idx] = out[1].detach().float().cpu()
        return hook

    for i, layer in enumerate(layers):
        h = layer.self_attn.register_forward_hook(make_attn_weight_hook(i))
        handles.append(h)

    try:
        with torch.no_grad():
            out = model(
                **inputs,
                output_attentions=True,   # only works with eager attention
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    num_layers = len(layers)
    num_heads  = model.config.text_config.num_attention_heads
    head_dim   = model.config.text_config.hidden_size // num_heads

    contrib_norms = np.zeros((num_layers, num_heads), dtype=np.float32)
    attn_entropy  = np.zeros((num_layers, num_heads), dtype=np.float32)
    attn_weights_list = []
    has_entropy = bool(attn_weight_captures)
    seq_len = 1

    for i, layer in enumerate(layers):
        # ---- per-head contribution norms (always available via hook) ----
        x = oproj_inputs[i]                        # (batch, seq, num_heads*head_dim)
        seq_len = x.shape[1]
        tok = token_idx if token_idx >= 0 else seq_len + token_idx
        x_tok = x[0, tok, :]                       # (num_heads * head_dim,)

        W = layer.self_attn.o_proj.weight.detach().float()  # (hidden, num_heads*head_dim)
        for h in range(num_heads):
            head_vec = x_tok[h * head_dim : (h + 1) * head_dim]  # (head_dim,)
            W_h      = W[:, h * head_dim : (h + 1) * head_dim]   # (hidden, head_dim)
            contrib  = head_vec @ W_h.T                           # (hidden,)
            contrib_norms[i, h] = contrib.norm().item()

        # ---- attention entropy (only when eager attention is used) ----
        if i in attn_weight_captures:
            aw = attn_weight_captures[i].numpy()   # (batch, num_heads, seq_q, seq_k)
            aw = aw[0]                             # (num_heads, seq_q, seq_k)
            attn_weights_list.append(aw)
            for h in range(num_heads):
                row = aw[h, tok, :]
                row = np.clip(row, 1e-9, None)
                attn_entropy[i, h] = -np.sum(row * np.log(row))

    result = {
        "contrib_norms": contrib_norms,
        "num_layers":    num_layers,
        "num_heads":     num_heads,
        "seq_len":       seq_len,
    }
    if has_entropy:
        result["attn_entropy"]  = attn_entropy
        result["attn_weights"]  = attn_weights_list
    else:
        result["attn_entropy"]  = None   # requires attn_implementation="eager"
        result["attn_weights"]  = []
        print("[head_analysis] Note: attention entropy unavailable with sdpa. "
              "Load model with attn_implementation='eager' to enable it.")
    return result


# ---------------------------------------------------------------------------
# Convenience wrapper that takes raw messages (like the server does)
# ---------------------------------------------------------------------------

def analyse_messages(model, processor, messages: list, token_idx: int = -1) -> dict:
    """
    High-level entry point.

    Args:
        messages: OpenAI-style list of dicts with 'role' and 'content'.
        token_idx: Token position to analyse (-1 = last).
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    result = get_head_activations(model, inputs, token_idx=token_idx)
    result["tokens"] = processor.tokenizer.tokenize(text)
    return result


# ---------------------------------------------------------------------------
# Comparison: two prompts
# ---------------------------------------------------------------------------

def compare_head_activations(result_a: dict, result_b: dict,
                              label_a: str = "A", label_b: str = "B") -> dict:
    """
    Compute element-wise difference in contribution norms between two prompts.

    Returns dict with:
        diff          : (num_layers, num_heads)  result_b - result_a
        ratio         : (num_layers, num_heads)  result_b / (result_a + eps)
        top_changed   : list of (layer, head, diff) sorted by |diff| descending
    """
    diff  = result_b["contrib_norms"] - result_a["contrib_norms"]
    ratio = result_b["contrib_norms"] / (result_a["contrib_norms"] + 1e-6)

    # Flatten and sort by absolute difference
    flat = [(int(l), int(h), float(diff[l, h]))
            for l in range(diff.shape[0])
            for h in range(diff.shape[1])]
    flat.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        "diff":        diff,
        "ratio":       ratio,
        "top_changed": flat,
        "label_a":     label_a,
        "label_b":     label_b,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_head_contrib_norms(result: dict, title: str = "Head Contribution Norms",
                             figsize=(14, 7)) -> plt.Figure:
    """Heatmap of (layer × head) contribution norms."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(result["contrib_norms"], aspect="auto", cmap="viridis")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="L2 norm of head contribution")
    plt.tight_layout()
    return fig


def plot_head_entropy(result: dict, title: str = "Attention Entropy (lower = more focused)",
                      figsize=(14, 7)) -> plt.Figure:
    """Heatmap of (layer × head) attention entropy. Requires eager attention mode."""
    if result.get("attn_entropy") is None:
        raise ValueError(
            "Entropy not available. Reload model with attn_implementation='eager'."
        )
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(result["attn_entropy"], aspect="auto", cmap="plasma_r")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    plt.tight_layout()
    return fig


def plot_comparison(cmp: dict, figsize=(14, 7)) -> plt.Figure:
    """Heatmap of contribution norm difference between two prompts."""
    diff = cmp["diff"]
    vmax = np.abs(diff).max()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Head norm diff: {cmp['label_b']} − {cmp['label_a']}")
    plt.colorbar(im, ax=ax, label="Δ L2 norm (blue=decreased, red=increased)")
    plt.tight_layout()
    return fig


def print_top_heads(cmp: dict, n: int = 20):
    """Print the top-N most changed heads between two prompts."""
    print(f"Top {n} heads changed: {cmp['label_b']} − {cmp['label_a']}")
    print(f"{'Layer':>6}  {'Head':>5}  {'Δnorm':>10}")
    print("-" * 28)
    for layer, head, delta in cmp["top_changed"][:n]:
        direction = "▲" if delta > 0 else "▼"
        print(f"{layer:>6}  {head:>5}  {delta:>+10.3f}  {direction}")
