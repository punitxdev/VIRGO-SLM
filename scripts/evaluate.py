#!/usr/bin/env python3
"""
Virgo SLM V1.0 - Evaluation & Perplexity Pipeline
Computes cross-entropy loss and perplexity across test validation datasets.
"""

import sys
import os
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.model import VirgoModel

def evaluate():
    print("=" * 60)
    print(" Virgo SLM Perplexity & Loss Evaluation")
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
    print("Model initialized for evaluation.")

if __name__ == "__main__":
    evaluate()
