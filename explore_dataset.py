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
DTYPE = "float16"  # "float16" for faster/less memory, "float32" for precision
MAX_TOKENS = 8192  # Gemma-2-2b context window

# Output directory
OUTPUT_DIR = "linebreak_data"

# %% Cell 2: Download and clean a single book for exploration

console.print("\n[bold cyan]═══ EXPLORING A SINGLE BOOK ═══[/bold cyan]\n")

book_id = BOOK_IDS[0]  # Pride and Prejudice
console.print(f"Downloading book {book_id}...")
text = download_gutenberg_book(book_id)
console.print(f"[green]✓[/green] Downloaded: {len(text):,} characters")

console.print("\nCleaning text...")
clean_text = clean_gutenberg_text(text)
paragraphs = [p for p in clean_text.split('\n\n') if len(p) >= MIN_SEQ_LENGTH]
console.print(f"[green]✓[/green] Cleaned text: {len(clean_text):,} characters")
console.print(f"[green]✓[/green] Found {len(paragraphs)} substantial paragraphs (≥{MIN_SEQ_LENGTH} chars)")

# %% Cell 3: Demonstrate wrapping at different widths

console.print("\n[bold cyan]═══ WRAPPING DEMONSTRATION ═══[/bold cyan]\n")

# Take first paragraph
sample_para = paragraphs[5] if len(paragraphs) > 5 else paragraphs[0]
sample_para = sample_para[:500]  # Keep it short for display

console.print("[yellow]Original paragraph (first 500 chars):[/yellow]")
console.print(f"{sample_para}\n")

demo_widths = [40, 80, 120]
console.print(f"[yellow]Wrapped at different widths:[/yellow]\n")

for width in demo_widths:
    wrapped = wrap_text_to_width(sample_para, width)
    console.print(f"[bold]Width {width}:[/bold]")
    console.print(wrapped)
    console.print()

# %% Cell 4: Create full dataset

console.print("\n[bold cyan]═══ CREATING FULL DATASET ═══[/bold cyan]\n")

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console
) as progress:
    
    # Download all books
    task1 = progress.add_task("[cyan]Downloading books...", total=len(BOOK_IDS))
    all_books = []
    for book_id in BOOK_IDS:
        text = download_gutenberg_book(book_id)
        clean = clean_gutenberg_text(text)
        all_books.append(clean)
        progress.update(task1, advance=1)
    
    # Extract paragraphs
    console.print("\n[cyan]Extracting paragraphs...[/cyan]")
    all_paragraphs = []
    for book_text in all_books:
        paras = [p for p in book_text.split('\n\n') if len(p) >= MIN_SEQ_LENGTH]
        all_paragraphs.extend(paras)
    
    # Create wrapped versions at each width
    task2 = progress.add_task("[cyan]Creating wrapped sequences...", total=len(LINE_WIDTHS))
    dataset = {}
    for width in LINE_WIDTHS:
        sequences = []
        for i in range(min(NUM_SEQUENCES, len(all_paragraphs))):
            para = all_paragraphs[i % len(all_paragraphs)]
            wrapped = wrap_text_to_width(para, width)
            sequences.append(wrapped)
        dataset[width] = sequences
        progress.update(task2, advance=1)

console.print(f"\n[green]✓[/green] Created dataset with {len(dataset)} widths")
console.print(f"  Total sequences: {sum(len(seqs) for seqs in dataset.values())}")

# %% Cell 5: Dataset statistics

console.print("\n[bold cyan]═══ DATASET STATISTICS ═══[/bold cyan]\n")

# Per-width statistics
console.print("[yellow]Per-width statistics:[/yellow]")
sample_widths = [15, 40, 80, 120, 150]
for width in sample_widths:
    if width in dataset:
        seqs = dataset[width]
        avg_len = sum(len(s) for s in seqs) / len(seqs)
        console.print(f"  Width {width:3d}: {len(seqs)} sequences, avg {avg_len:,.0f} chars")

# Overall statistics
console.print("\n[yellow]Overall statistics:[/yellow]")
first_width = sorted(dataset.keys())[0]
all_text = '\n\n'.join(dataset[first_width])
console.print(f"  Total unique sequences: {len(dataset[first_width])}")
console.print(f"  Total characters (unformatted): {len(all_text):,}")
console.print(f"  Unique characters: {len(set(all_text))}")

# Character distribution
from collections import Counter
char_counts = Counter(all_text)
console.print(f"\n[yellow]Most common characters:[/yellow]")
for char, count in char_counts.most_common(10):
    char_display = repr(char) if char in '\n\t ' else char
    pct = 100 * count / len(all_text)
    console.print(f"  {char_display}: {count:,} ({pct:.2f}%)")

# %% Cell 6: Save text files

console.print("\n[bold cyan]═══ SAVING TEXT FILES ═══[/bold cyan]\n")

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Saving text files...", total=len(LINE_WIDTHS))
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for width, sequences in dataset.items():
        output_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n=====\n\n'.join(sequences))
        progress.update(task, advance=1)

console.print(f"[green]✓[/green] Saved {len(dataset)} text files")

# %% Cell 7: Tokenize with Gemma-2-2b

console.print("\n[bold cyan]═══ TOKENIZING WITH GEMMA-2-2B ═══[/bold cyan]\n")

console.print("[yellow]Loading Gemma-2-2b tokenizer...[/yellow]")
tokenizer = get_gemma_tokenizer(device=DEVICE, dtype=DTYPE)
console.print("[green]✓[/green] Tokenizer loaded")

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Tokenizing sequences...", total=len(LINE_WIDTHS))
    
    tokenized_dataset = {}
    for width in LINE_WIDTHS:
        sequences = dataset[width]
        tokenized_sequences = []
        for seq in sequences:
            tokens = tokenizer.encode(seq, max_length=MAX_TOKENS, truncation=True)
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            tokenized_sequences.append(tokens)
        tokenized_dataset[width] = tokenized_sequences
        progress.update(task, advance=1)

console.print(f"[green]✓[/green] Tokenized {len(tokenized_dataset)} widths")

# Show sample tokens
sample_width = 40
if sample_width in tokenized_dataset:
    sample_tokens = tokenized_dataset[sample_width][0][:100]
    console.print(f"\n[yellow]Sample tokens (width {sample_width}, first 100):[/yellow]")
    console.print(f"  {len(sample_tokens)} tokens: {sample_tokens[:20]}...")
    decoded = tokenizer.decode(sample_tokens)
    console.print(f"  Decoded: {decoded[:150]}...")

# %% Cell 8: Generate metadata

console.print("\n[bold cyan]═══ GENERATING TOKEN METADATA ═══[/bold cyan]\n")

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Generating metadata...", total=len(LINE_WIDTHS))
    
    from linebreak_utils import generate_token_metadata
    metadata_dataset = {}
    
    for width in LINE_WIDTHS:
        sequences = dataset[width]
        tokenized_sequences = tokenized_dataset[width]
        metadata_sequences = []
        
        for text, tokens in zip(sequences, tokenized_sequences):
            metadata = generate_token_metadata(text, tokens, tokenizer, width)
            metadata_sequences.append(metadata)
        
        metadata_dataset[width] = metadata_sequences
        progress.update(task, advance=1)

console.print(f"[green]✓[/green] Generated metadata for {len(metadata_dataset)} widths")

# Show sample metadata
sample_width = 40
if sample_width in metadata_dataset:
    sample_meta = metadata_dataset[sample_width][0][:5]
    console.print(f"\n[yellow]Sample metadata (width {sample_width}, first 5 tokens):[/yellow]")
    for i, meta in enumerate(sample_meta):
        decoded_token = tokenizer.decode([tokenized_dataset[sample_width][0][i]])
        console.print(f"  Token {i}: '{decoded_token}'")
        console.print(f"    char_pos={meta['char_position']}, line_width={meta['line_width']}, "
                     f"chars_remaining={meta['chars_remaining']}")
        console.print(f"    token_len={meta['token_length']}, next_token_len={meta['next_token_length']}, "
                     f"line={meta['line_number']}, is_newline={meta['is_newline']}")

# %% Cell 9: Save tokenized data and metadata

console.print("\n[bold cyan]═══ SAVING TOKENIZED DATA & METADATA ═══[/bold cyan]\n")

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    
    # Save JSON and PyTorch tensors
    task = progress.add_task("[cyan]Saving tokens & metadata...", total=len(LINE_WIDTHS))
    
    import torch
    import json
    
    for width in LINE_WIDTHS:
        # JSON tokens
        json_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_tokens.json")
        with open(json_file, 'w') as f:
            json.dump(tokenized_dataset[width], f)
        
        # PyTorch tokens
        max_len = max(len(seq) for seq in tokenized_dataset[width])
        padded = [seq + [0] * (max_len - len(seq)) for seq in tokenized_dataset[width]]
        tokens_tensor = torch.tensor(padded, dtype=torch.long)
        pt_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_tokens.pt")
        torch.save(tokens_tensor, pt_file)
        
        # PyTorch metadata
        if metadata_dataset:
            metadata_tensor = []
            for seq_meta in metadata_dataset[width]:
                seq_tensor = [[m['char_position'], m['line_width'], m['chars_remaining'],
                              m['token_length'], m['next_token_length'], m['line_number'],
                              m['is_newline']] for m in seq_meta]
                # Pad to max_len
                while len(seq_tensor) < max_len:
                    seq_tensor.append([0, 0, 0, 0, 0, 0, 0])
                metadata_tensor.append(seq_tensor)
            metadata_tensor = torch.tensor(metadata_tensor, dtype=torch.long)
            meta_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_metadata.pt")
            torch.save(metadata_tensor, meta_file)
        
        progress.update(task, advance=1)

console.print(f"\n[bold green]✓ DATASET COMPLETE![/bold green]")
console.print(f"\nAll files saved to '[cyan]{OUTPUT_DIR}/[/cyan]'")
console.print(f"\n[bold]Files created:[/bold]")
console.print(f"  • linebreak_width_*.txt (raw text)")
console.print(f"  • linebreak_width_*_tokens.json (token IDs as JSON)")
console.print(f"  • linebreak_width_*_tokens.pt (token tensors)")
console.print(f"  • linebreak_width_*_metadata.pt (metadata tensors)")
console.print(f"\n[bold]Metadata features per token:[/bold]")
console.print(f"  [char_position, line_width, chars_remaining, token_length,")
console.print(f"   next_token_length, line_number, is_newline]")
console.print(f"\n[green]Ready for training![/green] 🚀")
