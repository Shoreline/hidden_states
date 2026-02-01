"""
Visualization utilities for hidden states analysis.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hidden_state_norms(hidden_states, layer_indices=None, title="Hidden State Norms"):
    """
    Plot the L2 norm of hidden states across layers.
    
    Args:
        hidden_states: Tuple of hidden states from model
        layer_indices: Optional list of layer indices to plot
        title: Plot title
    """
    if layer_indices is None:
        layer_indices = range(len(hidden_states))
    
    norms = []
    for i in layer_indices:
        hs = hidden_states[i]
        norm = torch.norm(hs, dim=-1).mean().item()
        norms.append(norm)
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, norms, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Mean L2 Norm")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


def plot_hidden_state_heatmap(hidden_state, token_labels=None, title="Hidden State Heatmap"):
    """
    Plot a heatmap of hidden state activations.
    
    Args:
        hidden_state: Single layer hidden state (batch, seq_len, hidden_dim)
        token_labels: Optional labels for tokens
        title: Plot title
    """
    # Take first batch, reduce hidden dim by averaging chunks
    hs = hidden_state[0].cpu().numpy()  # (seq_len, hidden_dim)
    
    # Reduce hidden dim for visualization (take every 100th dim or average)
    chunk_size = max(1, hs.shape[1] // 100)
    hs_reduced = hs[:, ::chunk_size]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(hs_reduced, cmap='RdBu_r', center=0)
    
    if token_labels:
        plt.yticks(range(len(token_labels)), token_labels)
    
    plt.xlabel("Hidden Dimension (sampled)")
    plt.ylabel("Token Position")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_layer_similarity(hidden_states, title="Layer-wise Cosine Similarity"):
    """
    Plot cosine similarity between consecutive layers.
    
    Args:
        hidden_states: Tuple of hidden states from model
        title: Plot title
    """
    similarities = []
    
    for i in range(len(hidden_states) - 1):
        hs1 = hidden_states[i][0].flatten()  # Flatten first batch
        hs2 = hidden_states[i + 1][0].flatten()
        
        sim = torch.nn.functional.cosine_similarity(
            hs1.unsqueeze(0), hs2.unsqueeze(0)
        ).item()
        similarities.append(sim)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(similarities)), similarities, marker='o')
    plt.xlabel("Layer Transition (i -> i+1)")
    plt.ylabel("Cosine Similarity")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


def compare_hidden_states(hs1, hs2, labels=("Input 1", "Input 2")):
    """
    Compare hidden states from two different inputs.
    
    Args:
        hs1: Hidden states from first input
        hs2: Hidden states from second input
        labels: Labels for the two inputs
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot norms comparison
    norms1 = [torch.norm(h, dim=-1).mean().item() for h in hs1]
    norms2 = [torch.norm(h, dim=-1).mean().item() for h in hs2]
    
    axes[0].plot(norms1, label=labels[0], marker='o')
    axes[0].plot(norms2, label=labels[1], marker='s')
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean L2 Norm")
    axes[0].set_title("Hidden State Norms Comparison")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot cosine similarity between the two
    similarities = []
    min_layers = min(len(hs1), len(hs2))
    for i in range(min_layers):
        # Compare mean hidden states
        mean1 = hs1[i].mean(dim=1).flatten()
        mean2 = hs2[i].mean(dim=1).flatten()
        
        sim = torch.nn.functional.cosine_similarity(
            mean1.unsqueeze(0), mean2.unsqueeze(0)
        ).item()
        similarities.append(sim)
    
    axes[1].plot(range(min_layers), similarities, marker='o', color='green')
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title(f"Similarity between {labels[0]} and {labels[1]}")
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig
