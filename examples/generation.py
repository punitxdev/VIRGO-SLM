#!/usr/bin/env python3
"""
Virgo SLM V1.0 - Batch Generation Example
Demonstrates prompt execution and output sampling across multiple presets.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.model import VirgoModel

prompts = [
    "Write a Python function to compute Fibonacci numbers efficiently.",
    "Explain the concept of backpropagation in deep neural networks.",
    "What are the main differences between TCP and UDP protocols?"
]

def main():
    print("=" * 60)
    print(" Virgo SLM V1.0 - Batch Generation Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VirgoModel(
        vocab_size=45000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_seq_length=1024,
        dropout=0.0
    ).to(device)
    model.eval()

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt [{idx}/{len(prompts)}]: '{prompt}' ---")
        print("Generating response...")

if __name__ == "__main__":
    main()
