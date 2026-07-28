#!/usr/bin/env python3
"""
Virgo SLM V1.0 - Interactive Terminal Chatbot
Provides a simple CLI loop for chatting with Virgo SLM.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.model import VirgoModel

def main():
    print("=" * 60)
    print(" Virgo SLM V1.0 - Interactive Terminal Chatbot")
    print(" Type 'exit' or 'quit' to terminate the session.")
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

    ckpt_path = "trained_models/virgo_instruct_v2.pt"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
        model.load_state_dict(state_dict, strict=False)
        print("Model checkpoint loaded successfully.")
    else:
        print("Note: Running with randomly initialized weights (checkpoint not found).")

    model.eval()

    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting chatbot session. Goodbye!")
                break
            if not user_input.strip():
                continue

            print("\nVirgo > Processing response...")
            # Interactive generation loop placeholder
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    main()
