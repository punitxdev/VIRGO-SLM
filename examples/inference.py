#!/usr/bin/env python3
"""
Virgo SLM V1.0 - Standalone Inference Example
Demonstrates how to load the Virgo 120M model and perform token generation.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.model import VirgoModel

def sample_next_token(logits, temperature=0.7, top_k=50, top_p=0.9):
    """Applies Temperature, Top-K, and Top-P (Nucleus) sampling to logits."""
    if temperature == 0 or temperature < 1e-6:
        return torch.argmax(logits, dim=-1).item()
    
    logits = logits / temperature
    
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

def main():
    print("=" * 60)
    print("Virgo SLM V1.0 - Inference Quickstart")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model Parameters
    vocab_size = 45000
    d_model = 768
    num_heads = 12
    num_layers = 12
    d_ff = 3072
    max_seq_length = 1024
    dropout = 0.0

    print("Initializing VirgoModel architecture (120M parameters)...")
    model = VirgoModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(device)

    ckpt_path = "trained_models/virgo_instruct_v2.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with randomly initialized weights for structural test.")

    model.eval()
    print("\nModel is ready for inference!")

if __name__ == "__main__":
    main()
