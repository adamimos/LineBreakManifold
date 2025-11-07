"""
Interactive exploration of the linebreaking dataset.

This notebook-style script demonstrates the dataset curation pipeline for the
linebreaking task from "When Models Manipulate Manifolds" (Anthropic).

Run cells individually in VSCode or similar IDEs that support #%% cell markers.
"""

# %% Cell 1: Import utilities and set parameters

from linebreak_utils import (
    download_gutenberg_book,
    clean_gutenberg_text,
    wrap_text_to_width,
    create_linebreak_dataset,
    save_dataset,
    get_gemma_tokenizer,
    tokenize_dataset,
    create_metadata_dataset,
    save_tokenized_dataset,
    save_tokenized_dataset_torch
)
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console

console = Console()

# Parameters for dataset creation
BOOK_IDS = [
    1342,  # Pride and Prejudice - Jane Austen
    11,    # Alice's Adventures in Wonderland - Lewis Carroll
    1661,  # The Adventures of Sherlock Holmes - Arthur Conan Doyle
    84,    # Frankenstein - Mary Shelley
    2701,  # Moby Dick - Herman Melville
    1952,  # The Yellow Wallpaper - Charlotte Perkins Gilman
    158,   # Emma - Jane Austen
    345,   # Dracula - Bram Stoker
    98,    # A Tale of Two Cities - Charles Dickens
    174,   # The Picture of Dorian Gray - Oscar Wilde
]

# Line widths to generate (following paper's methodology: k=15,20,...,150)
LINE_WIDTHS = list(range(15, 151, 5))  # [15, 20, 25, 30, ..., 145, 150]

# Target number of sequences
NUM_SEQUENCES = 200

# Minimum sequence length (just need substantial content)
# You can truncate to fit your model's context window during training
MIN_SEQ_LENGTH = 500

# Gemma-2-2b settings
DEVICE = "mps"  # Use "cuda" for NVIDIA GPUs, "cpu" for CPU
DTYPE = "float16"  # Faster and uses less memory
MAX_TOKENS = 8192  # Gemma-2-2b context window (vs GPT-2's 1024)

print("✓ Imports successful")
print(f"✓ Will process {len(BOOK_IDS)} books")
print(f"✓ Target: {NUM_SEQUENCES} sequences (min {MIN_SEQ_LENGTH} chars)")
print(f"✓ Line widths: {len(LINE_WIDTHS)} widths from {LINE_WIDTHS[0]} to {LINE_WIDTHS[-1]}")
print(f"✓ Model: Gemma-2-2b (device={DEVICE}, max_tokens={MAX_TOKENS})")

# %% Cell 2: Download and clean a single book, show statistics

print("=" * 60)
print("DOWNLOADING SAMPLE BOOK: Pride and Prejudice")
print("=" * 60)

# Download and clean a single book for demonstration
sample_book_id = 1342  # Pride and Prejudice
raw_text = download_gutenberg_book(sample_book_id)
cleaned_text = clean_gutenberg_text(raw_text)

# Show statistics
print(f"\nRaw text length: {len(raw_text):,} characters")
print(f"Cleaned text length: {len(cleaned_text):,} characters")
print(f"Reduction: {100 * (1 - len(cleaned_text)/len(raw_text)):.1f}%")

paragraphs = cleaned_text.split('\n\n')
print(f"\nParagraph count: {len(paragraphs)}")
print(f"Average paragraph length: {len(cleaned_text) // len(paragraphs)} characters")

# Show first paragraph as example
if paragraphs:
    print("\n" + "-" * 60)
    print("FIRST PARAGRAPH:")
    print("-" * 60)
    print(paragraphs[0][:300] + "..." if len(paragraphs[0]) > 300 else paragraphs[0])
else:
    print("\n⚠ Warning: No paragraphs found after cleaning!")

# %% Cell 3: Demonstrate wrapping at different widths (side-by-side comparison)

print("\n" + "=" * 60)
print("WRAPPING DEMONSTRATION")
print("=" * 60)

# Select a paragraph for demonstration
# Use a paragraph that exists (fallback to first if not enough paragraphs)
para_index = min(5, len(paragraphs) - 1) if paragraphs else 0
demo_paragraph = paragraphs[para_index] if paragraphs else "Sample text for demonstration."
demo_widths = [40, 80, 120]

print(f"\nOriginal paragraph ({len(demo_paragraph)} chars):")
print("-" * 60)
print(demo_paragraph[:200] + "..." if len(demo_paragraph) > 200 else demo_paragraph)

for width in demo_widths:
    wrapped = wrap_text_to_width(demo_paragraph, width)
    lines = wrapped.split('\n')
    
    print(f"\n{'=' * 60}")
    print(f"WIDTH = {width} characters ({len(lines)} lines)")
    print('=' * 60)
    print(wrapped)

# %% Cell 4: Create full dataset from multiple books

print("\n" + "=" * 80)
print("CREATING FULL DATASET")
print("=" * 80)

# Create the complete dataset
dataset = create_linebreak_dataset(
    book_ids=BOOK_IDS,
    num_sequences=NUM_SEQUENCES,
    line_widths=LINE_WIDTHS,
    min_seq_length=MIN_SEQ_LENGTH
)

print("\n✓ Dataset creation complete!")

# %% Cell 5: Show dataset statistics

print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

for width in sorted(dataset.keys()):
    sequences = dataset[width]
    
    print(f"\nWidth {width}:")
    print(f"  Sequences: {len(sequences)}")
    
    # Calculate statistics
    total_chars = sum(len(seq) for seq in sequences)
    avg_length = total_chars // len(sequences) if sequences else 0
    
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average sequence length: {avg_length:,} chars")
    
    # Count lines
    total_lines = sum(seq.count('\n') + 1 for seq in sequences)
    avg_lines = total_lines / len(sequences) if sequences else 0
    print(f"  Total lines: {total_lines:,}")
    print(f"  Average lines per sequence: {avg_lines:.1f}")
    
    # Show sample (first 200 chars of first sequence)
    if sequences:
        sample = sequences[0][:200]
        print(f"  Sample (first sequence):")
        print(f"    {sample}...")

# Character distribution across entire dataset
print("\n" + "-" * 60)
print("OVERALL STATISTICS:")
print("-" * 60)

# Use the first width as representative (content is the same, just formatted differently)
first_width = sorted(dataset.keys())[0]
all_text = '\n\n'.join(dataset[first_width])

print(f"Total unique sequences: {len(dataset[first_width])}")
print(f"Total characters (unformatted): {len(all_text):,}")
print(f"Unique characters: {len(set(all_text))}")

# Most common characters
from collections import Counter
char_counts = Counter(all_text)
print(f"\nMost common characters:")
for char, count in char_counts.most_common(10):
    char_display = repr(char) if char in '\n\t ' else char
    print(f"  {char_display}: {count:,} ({100*count/len(all_text):.2f}%)")

# %% Cell 6: Save text dataset to disk

OUTPUT_DIR = "linebreak_data"

print("\n" + "=" * 60)
print("SAVING TEXT DATASET")
print("=" * 60)

save_dataset(dataset, OUTPUT_DIR)

# %% Cell 7: Tokenize dataset with Gemma-2-2b tokenizer

print("\n" + "=" * 80)
print("TOKENIZING DATASET WITH GEMMA-2-2B")
print("=" * 80)

# Get the tokenizer
tokenizer = get_gemma_tokenizer(device=DEVICE, dtype=DTYPE)

# Tokenize all sequences (truncate to 8192 tokens for Gemma-2-2b)
tokenized_dataset = tokenize_dataset(
    dataset, 
    tokenizer, 
    max_length=MAX_TOKENS,
    device=DEVICE,
    dtype=DTYPE
)

print("\n" + "-" * 60)
print("TOKENIZATION STATISTICS:")
print("-" * 60)

# Show token statistics for a few widths
for width in [15, 40, 80, 120, 150]:
    if width in tokenized_dataset:
        token_seqs = tokenized_dataset[width]
        lengths = [len(seq) for seq in token_seqs]
        avg_len = sum(lengths) // len(lengths)
        
        print(f"\nWidth {width}:")
        print(f"  Sequences: {len(token_seqs)}")
        print(f"  Avg tokens: {avg_len}")
        print(f"  Token range: {min(lengths)}-{max(lengths)}")
        
        # Show a decoded sample
        if width == 40:
            sample_tokens = token_seqs[0][:100]  # First 100 tokens
            decoded = tokenizer.decode(sample_tokens)
            print(f"  Sample (first 100 tokens decoded):")
            print(f"    {decoded[:150]}...")

# %% Cell 8: Generate token metadata

print("\n" + "=" * 80)
print("GENERATING TOKEN METADATA")
print("=" * 80)

# Generate metadata for each token position
metadata_dataset = create_metadata_dataset(
    dataset,
    tokenized_dataset,
    tokenizer,
    device=DEVICE,
    dtype=DTYPE
)

# Show sample metadata
print("\n" + "-" * 60)
print("SAMPLE METADATA (Width 40, First Sequence, First 5 Tokens):")
print("-" * 60)

sample_width = 40
sample_metadata = metadata_dataset[sample_width][0][:5]
sample_tokens = tokenized_dataset[sample_width][0][:5]

for i, (token_id, meta) in enumerate(zip(sample_tokens, sample_metadata)):
    token_str = tokenizer.decode([token_id])
    print(f"\nToken {i}: {repr(token_str)} (ID: {token_id})")
    print(f"  Char Position: {meta['char_position']}/{meta['line_width']}")
    print(f"  Chars Remaining: {meta['chars_remaining']}")
    print(f"  Token Length: {meta['token_length']} chars")
    print(f"  Next Token Length: {meta['next_token_length']} chars")
    print(f"  Line Number: {meta['line_number']}")
    print(f"  Is Newline: {meta['is_newline']}")

# %% Cell 9: Save tokenized dataset with metadata

print("\n" + "=" * 60)
print("SAVING TOKENIZED DATASET WITH METADATA")
print("=" * 60)

# Save as JSON (human-readable, easy to load)
save_tokenized_dataset(tokenized_dataset, OUTPUT_DIR)

# Save as PyTorch tensors with metadata (efficient for training)
save_tokenized_dataset_torch(tokenized_dataset, OUTPUT_DIR, metadata_dataset)

print(f"\n✓ All done! Dataset ready at ./{OUTPUT_DIR}/")
print(f"\nFiles created:")
print(f"  - linebreak_width_*.txt (raw text)")
print(f"  - linebreak_width_*_tokens.json (token IDs as JSON)")
print(f"  - linebreak_width_*_tokens.pt (token IDs as PyTorch tensors)")
print(f"  - linebreak_width_*_metadata.pt (token metadata tensors)")
print(f"\nMetadata features per token:")
print(f"  [char_position, line_width, chars_remaining, token_length,")
print(f"   next_token_length, line_number, is_newline]")
print(f"\nYou can now use these files to train or evaluate models on the")
print(f"linebreaking task, following the methodology from the Anthropic paper.")

