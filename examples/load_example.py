"""
Example: How to load and use the tokenized datasets
"""

import json
import torch

# Example 1: Load JSON tokens
print("="*60)
print("LOADING JSON TOKENS")
print("="*60)

with open('linebreak_data/linebreak_width_40_tokens.json', 'r') as f:
    tokens_40 = json.load(f)

print(f"Loaded {len(tokens_40)} sequences")
print(f"First sequence has {len(tokens_40[0])} tokens")
print(f"First 20 tokens: {tokens_40[0][:20]}")

# Example 2: Load PyTorch tensors
print("\n" + "="*60)
print("LOADING PYTORCH TENSORS")
print("="*60)

tokens_tensor = torch.load('linebreak_data/linebreak_width_40_tokens.pt')
print(f"Tensor shape: {tokens_tensor.shape}")  # [200, 1024]
print(f"Data type: {tokens_tensor.dtype}")
print(f"First sequence (first 20 tokens):\n{tokens_tensor[0, :20]}")

# Example 3: Decode back to text
print("\n" + "="*60)
print("DECODING TOKENS BACK TO TEXT")
print("="*60)

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
tokenizer = model.tokenizer

# Decode first sequence
decoded_text = tokenizer.decode(tokens_40[0])
print(f"Decoded text (first 200 chars):")
print(decoded_text[:200])

# Example 4: Compare different widths
print("\n" + "="*60)
print("COMPARING DIFFERENT WIDTHS")
print("="*60)

with open('linebreak_data/linebreak_width_15_tokens.json', 'r') as f:
    tokens_15 = json.load(f)

with open('linebreak_data/linebreak_width_150_tokens.json', 'r') as f:
    tokens_150 = json.load(f)

print(f"Width 15:  {len(tokens_15[0])} tokens")
print(f"Width 40:  {len(tokens_40[0])} tokens")
print(f"Width 150: {len(tokens_150[0])} tokens")

print("\nDecoded samples (first 100 chars):")
print(f"Width 15:\n{tokenizer.decode(tokens_15[0][:50])}\n")
print(f"Width 150:\n{tokenizer.decode(tokens_150[0][:50])}\n")

print("="*60)
print("✓ All examples completed!")
print("="*60)

