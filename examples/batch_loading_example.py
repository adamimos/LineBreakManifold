"""
Example: How to load batches of tokens and metadata for training

Shows both manual batch loading and PyTorch DataLoader usage.
"""

import torch
from linebreak_utils import load_linebreak_batch, create_dataloader

print("="*70)
print("METHOD 1: MANUAL BATCH LOADING")
print("="*70)

# Load a batch manually
batch = load_linebreak_batch(
    data_dir="linebreak_data",
    widths=[40, 80, 120],        # Load from 3 different widths
    batch_size=16,                # 16 sequences per width = 48 total
    max_length=1024,              # Truncate to 1024 tokens
    device="mps"                  # Use Metal Performance Shaders on M3
)

print(f"\nBatch contents:")
print(f"  Tokens shape: {batch['tokens'].shape}")        # [48, 1024]
print(f"  Metadata shape: {batch['metadata'].shape}")    # [48, 1024, 7]
print(f"  Width labels: {batch['widths'].shape}")        # [48]

print(f"\nWidth distribution:")
for width in [40, 80, 120]:
    count = (batch['widths'] == width).sum().item()
    print(f"  Width {width}: {count} sequences")

print(f"\nExample metadata for first sequence, first token:")
first_meta = batch['metadata'][0, 0]
print(f"  char_position: {first_meta[0].item()}")
print(f"  line_width: {first_meta[1].item()}")
print(f"  chars_remaining: {first_meta[2].item()}")
print(f"  token_length: {first_meta[3].item()}")
print(f"  next_token_length: {first_meta[4].item()}")
print(f"  line_number: {first_meta[5].item()}")
print(f"  is_newline: {first_meta[6].item()}")

print("\n" + "="*70)
print("METHOD 2: PYTORCH DATALOADER")
print("="*70)

# Create a dataloader
dataloader = create_dataloader(
    data_dir="linebreak_data",
    widths=[40, 80, 120],
    batch_size=32,
    max_length=1024,
    shuffle=True,
    device="cpu"  # DataLoader handles device transfer
)

print(f"\nDataLoader created with {len(dataloader.dataset)} total sequences")
print(f"Number of batches: {len(dataloader)}")

# Iterate through batches
print("\nIterating through first 3 batches:")
for i, batch in enumerate(dataloader):
    if i >= 3:
        break
    
    tokens = batch['tokens']
    metadata = batch['metadata']
    widths = batch['widths']
    
    print(f"\nBatch {i+1}:")
    print(f"  Shape: tokens={tokens.shape}, metadata={metadata.shape}")
    print(f"  Widths in batch: {widths.unique().tolist()}")
    
    # Example: Move to device for training
    if torch.backends.mps.is_available():
        tokens = tokens.to('mps')
        metadata = metadata.to('mps')
        print(f"  Moved to MPS device")

print("\n" + "="*70)
print("TRAINING LOOP EXAMPLE")
print("="*70)

print("""
# Example training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get data
        tokens = batch['tokens'].to(device)
        metadata = batch['metadata'].to(device)
        widths = batch['widths'].to(device)
        
        # Forward pass
        logits, newline_predictions = model(tokens, metadata)
        
        # Compute losses
        lm_loss = criterion(logits, targets)
        newline_loss = newline_criterion(newline_predictions, newline_labels)
        
        total_loss = lm_loss + newline_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
""")

print("\n" + "="*70)
print("ADVANCED: LOADING SPECIFIC WIDTHS FOR EVALUATION")
print("="*70)

# Load only narrow widths for testing
narrow_batch = load_linebreak_batch(
    data_dir="linebreak_data",
    widths=[15, 20, 25],
    batch_size=8,
    max_length=512,
    device="cpu"
)

print(f"\nNarrow width batch:")
print(f"  Sequences: {narrow_batch['tokens'].shape[0]}")
print(f"  Max line width: {narrow_batch['widths'].max().item()}")
print(f"  Min line width: {narrow_batch['widths'].min().item()}")

# Load only wide widths
wide_batch = load_linebreak_batch(
    data_dir="linebreak_data",
    widths=[130, 140, 150],
    batch_size=8,
    max_length=512,
    device="cpu"
)

print(f"\nWide width batch:")
print(f"  Sequences: {wide_batch['tokens'].shape[0]}")
print(f"  Max line width: {wide_batch['widths'].max().item()}")
print(f"  Min line width: {wide_batch['widths'].min().item()}")

print("\n✓ Ready for training with efficient batch loading!")

