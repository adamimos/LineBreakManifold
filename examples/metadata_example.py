"""
Example: How to use token metadata for the linebreaking task

The metadata tells the model WHERE it is on each line and helps it decide
when to insert line breaks based on the width constraint.
"""

import torch
import json

print("="*70)
print("LOADING TOKENIZED DATA WITH METADATA")
print("="*70)

# Load tokens (input to the model)
tokens = torch.load('linebreak_data/linebreak_width_40_tokens.pt')
print(f"\nTokens shape: {tokens.shape}")  # [num_sequences, max_length]

# Load metadata (additional features for each token)
metadata = torch.load('linebreak_data/linebreak_width_40_metadata.pt')
print(f"Metadata shape: {metadata.shape}")  # [num_sequences, max_length, 7]

# Metadata features:
# [0] char_position - Where we are in the current line (0 to line_width)
# [1] line_width - The constraint k (40 in this case)
# [2] chars_remaining - How much space left (line_width - char_position)
# [3] token_length - Size of current token in characters
# [4] next_token_length - Size of next token in characters
# [5] line_number - Which line we're on
# [6] is_newline - 1 if this token is/contains a newline, 0 otherwise

print("\n" + "="*70)
print("EXAMPLE: First sequence, first 10 tokens")
print("="*70)

# Get first sequence
seq_tokens = tokens[0, :10]
seq_metadata = metadata[0, :10]

print(f"\n{'Token ID':<10} {'Pos/Width':<12} {'Remaining':<10} {'Tok Len':<8} {'Next Len':<8} {'Line':<6} {'Newline'}")
print("-"*70)

for i in range(10):
    token_id = seq_tokens[i].item()
    char_pos = seq_metadata[i, 0].item()
    line_width = seq_metadata[i, 1].item()
    chars_rem = seq_metadata[i, 2].item()
    tok_len = seq_metadata[i, 3].item()
    next_len = seq_metadata[i, 4].item()
    line_num = seq_metadata[i, 5].item()
    is_newline = seq_metadata[i, 6].item()
    
    print(f"{token_id:<10} {char_pos}/{line_width:<9} {chars_rem:<10} {tok_len:<8} {next_len:<8} {line_num:<6} {bool(is_newline)}")

print("\n" + "="*70)
print("USE CASES FOR TRAINING")
print("="*70)

print("""
1. **Predicting Line Breaks**: The model can use:
   - chars_remaining: Is there room for the next token?
   - next_token_length: Will the next token fit?
   - char_position: Are we near the line boundary?

2. **Auxiliary Tasks**: Train the model to predict:
   - When to insert a newline (is_newline = 1)
   - How much space is left (chars_remaining)
   - Where we are on the line (char_position)

3. **Conditioning**: Give the model metadata as input features:
   - Concatenate metadata with token embeddings
   - Use as positional encodings
   - Use in attention mechanisms

4. **Evaluation**: Measure if the model learns:
   - To break near line_width
   - Word boundaries (not mid-word)
   - Consistent formatting across widths
""")

print("\n" + "="*70)
print("EXAMPLE TRAINING SETUP")
print("="*70)

print("""
```python
import torch.nn as nn

class LineBreakTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Encode metadata as additional features
        self.metadata_encoder = nn.Linear(7, d_model)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=6
        )
        
        # Predict next token + whether to insert newline
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.newline_head = nn.Linear(d_model, 1)
    
    def forward(self, tokens, metadata):
        # Embed tokens
        token_embeds = self.embedding(tokens)
        
        # Encode metadata
        metadata_embeds = self.metadata_encoder(metadata.float())
        
        # Combine token and metadata information
        combined = token_embeds + metadata_embeds
        
        # Process with transformer
        output = self.transformer(combined)
        
        # Predict next token and newline decision
        logits = self.lm_head(output)
        newline_logits = self.newline_head(output)
        
        return logits, newline_logits
```
""")

print("\n✓ Ready for training with rich positional features!")

