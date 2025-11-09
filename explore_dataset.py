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
    get_gemma_tokenizer
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

# Gemma-2-27b settings
MAX_TOKENS = 8192  # Gemma-2 context window

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
    from linebreak_utils import validate_wrapped_text, paragraph_has_word_exceeding_width

    for width in LINE_WIDTHS:
        sequences = []
        # Iterate deterministically through paragraphs and pick those safe for this width
        for para in all_paragraphs:
            if paragraph_has_word_exceeding_width(para, width):
                continue  # skip paragraphs that would force an overlong line
            wrapped = wrap_text_to_width(para, width)
            # Validate aggressively as a safety net
            assert validate_wrapped_text(wrapped, width), f"Found line >{width} for width {width}"
            sequences.append(wrapped)
            if len(sequences) >= NUM_SEQUENCES:
                break
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

# %% Cell 7: Tokenize with Gemma-2-27b

console.print("\n[bold cyan]═══ TOKENIZING WITH GEMMA-2-27B ═══[/bold cyan]\n")

console.print("[yellow]Loading Gemma-2 tokenizer...[/yellow]")
tokenizer = get_gemma_tokenizer()
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

# %% Cell 8: Compute line features (char_count, contains_newline)

console.print("\n[bold cyan]═══ COMPUTING LINE FEATURES ═══[/bold cyan]\n")

def compute_line_features_per_token(text, token_ids, tokenizer, max_length=None):
    """Compute per-token features using offsets into the wrapped text when possible.
    - char_count: characters since the most recent newline after consuming the token
    - contains_newline: 1 if the token's span contains '\\n', else 0
    Falls back to per-token decode if offsets are unavailable; in fallback mode,
    ignores leading spaces immediately after a newline to avoid overcounting.
    """
    # Try robust offsets-based path
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            max_length=(max_length if max_length else len(token_ids)),
            truncation=True,
        )
        # Handle batch vs single outputs
        input_ids = enc["input_ids"][0] if isinstance(enc["input_ids"][0], list) else enc["input_ids"]
        offsets = enc["offset_mapping"][0] if isinstance(enc["offset_mapping"][0], list) else enc["offset_mapping"]

        if len(input_ids) != len(token_ids):
            raise ValueError("Tokenizer mismatch with provided token_ids")

        # Precompute last newline index at each position
        n = len(text)
        last_nl = [-1] * (n + 1)
        last = -1
        for i, ch in enumerate(text):
            if ch == "\n":
                last = i
            last_nl[i + 1] = last

        char_counts = []
        contains_newlines = []
        for (start, end) in offsets:
            # Some special tokens can have (0,0); keep last count if exists
            if start == end == 0 and char_counts:
                char_counts.append(char_counts[-1])
                contains_newlines.append(0)
                continue

            span = text[start:end]
            contains_newlines.append(1 if "\n" in span else 0)

            # Distance from last newline before end to end
            ln_idx = last_nl[end]
            count = end - (ln_idx + 1)
            if count < 0:
                count = 0
            char_counts.append(count)

        return char_counts, contains_newlines

    except Exception:
        # Fallback: per-token decode, ignore leading spaces at line start
        char_counts = []
        contains_newlines = []
        line_count = 0
        at_line_start = True

        for tok in token_ids:
            piece = tokenizer.decode([tok]).replace("\r\n", "\n").replace("\r", "\n")
            contains_newlines.append(1 if "\n" in piece else 0)
            for ch in piece:
                if ch == "\n":
                    line_count = 0
                    at_line_start = True
                else:
                    if at_line_start and ch == " ":
                        # ignore synthetic leading spaces at line start
                        continue
                    at_line_start = False
                    line_count += 1
            char_counts.append(line_count)

        return char_counts, contains_newlines

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Computing line features...", total=len(LINE_WIDTHS))
    
    char_count_dataset = {}
    contains_newline_dataset = {}
    
    for width in LINE_WIDTHS:
        texts = dataset[width]
        tokenized_sequences = tokenized_dataset[width]
        char_count_sequences = []
        contains_newline_sequences = []
        
        for text, tokens in zip(texts, tokenized_sequences):
            counts, flags = compute_line_features_per_token(text, tokens, tokenizer, max_length=MAX_TOKENS)
            char_count_sequences.append(counts)
            contains_newline_sequences.append(flags)
        
        char_count_dataset[width] = char_count_sequences
        contains_newline_dataset[width] = contains_newline_sequences
        progress.update(task, advance=1)

console.print(f"[green]✓[/green] Computed line features for {len(char_count_dataset)} widths")

# Show sample features
sample_width = 40
if sample_width in char_count_dataset:
    sample_counts = char_count_dataset[sample_width][0][:10]
    sample_flags = contains_newline_dataset[sample_width][0][:10]
    console.print(f"\n[yellow]Sample features (width {sample_width}, first 10 tokens):[/yellow]")
    console.print(f"  char_count: {sample_counts}")
    console.print(f"  contains_newline: {sample_flags}")

# %% Cell 9: Save tokenized data and line features

console.print("\n[bold cyan]═══ SAVING TOKENIZED DATA & LINE FEATURES ═══[/bold cyan]\n")

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    
    # Save JSON and PyTorch tensors
    task = progress.add_task("[cyan]Saving tokens & line features...", total=len(LINE_WIDTHS))
    
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

        # PyTorch line features
        # char_count
        char_padded = [seq + [0] * (max_len - len(seq)) for seq in char_count_dataset[width]]
        char_tensor = torch.tensor(char_padded, dtype=torch.long)
        char_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_char_count.pt")
        torch.save(char_tensor, char_file)

        # contains_newline
        contains_padded = [seq + [0] * (max_len - len(seq)) for seq in contains_newline_dataset[width]]
        contains_tensor = torch.tensor(contains_padded, dtype=torch.long)
        contains_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_contains_newline.pt")
        torch.save(contains_tensor, contains_file)
        
        progress.update(task, advance=1)

console.print(f"\n[bold green]✓ DATASET COMPLETE![/bold green]")
console.print(f"\nAll files saved to '[cyan]{OUTPUT_DIR}/[/cyan]'")
console.print(f"\n[bold]Files created:[/bold]")
console.print(f"  • linebreak_width_*.txt (raw text)")
console.print(f"  • linebreak_width_*_tokens.json (token IDs as JSON)")
console.print(f"  • linebreak_width_*_tokens.pt (token tensors)")
console.print(f"  • linebreak_width_*_char_count.pt (chars since last newline per token)")
console.print(f"  • linebreak_width_*_contains_newline.pt (1 if token contains '\\n')")
console.print(f"\n[green]Ready for analysis![/green] 🚀")
