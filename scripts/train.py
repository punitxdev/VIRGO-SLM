#!/usr/bin/env python3
"""
Virgo SLM V1.0 - PyTorch Training Script
Pre-training and fine-tuning entry point for the Virgo 120M Transformer.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.model import VirgoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train Virgo SLM Model")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    return parser.parse_args()

def train():
    args = parse_args()
    print("=" * 60)
    print(f" Starting Virgo SLM Training Phase")
    print(f" Learning Rate: {args.lr} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
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

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training loop ready.")

if __name__ == "__main__":
    train()
