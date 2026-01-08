# %% [markdown]
# Colab: Linebreak Dataset + Model Analysis
#
# This single-file script merges the functionality of `explore_dataset.py`
# and `gpt2_small_cell.py` into VS Code-style cells so you can paste/run in
# Google Colab or VS Code Python Interactive. It:
# - Downloads and cleans Project Gutenberg books
# - Wraps text to multiple fixed line widths (word-boundary wrapping)
# - Tokenizes with a Hugging Face tokenizer (GPT-2 by default)
# - Computes per-token line features (chars since last newline, newline flag)
# - Saves text, tokens, and features to `linebreak_data/`
# - Streams sequences at the model context window and aggregates activations
# - Runs a 3D PCA of per-char_count mean activations and visualizes with Plotly
#
# Notes for Colab:
# - Default tokenizer/model are small and ungated (GPT-2) for easy demo.
# - To use Gemma-2-27b, accept its license on Hugging Face and set MODEL_ID
#   to "google/gemma-2-27b" and TOKENIZER_ID to the same, then login.
# - This script installs missing packages automatically in a cell.

# %%
# Setup: install any missing packages (minimal impact on Colab runtime)
import importlib, sys, subprocess

def ensure_packages(mod_to_pip):
    missing = []
    for mod, pip_name in mod_to_pip:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
    if missing:
        print(f"Installing: {' '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

ensure_packages([
    ("requests", "requests"),
    ("rich", "rich"),
    ("transformers", "transformers"),
    ("transformer_lens", "transformer-lens"),
    ("sklearn", "scikit-learn"),
    ("plotly", "plotly"),
])

# %%
# Imports and global configuration
import os
import re
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Iterator

import requests
import torch
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn, BarColumn, TextColumn
from rich.console import Console
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

console = Console()

# Data settings (same spirit as explore_dataset.py)
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

LINE_WIDTHS = list(range(15, 151, 5))
NUM_SEQUENCES = 200
MIN_SEQ_LENGTH = 500

# Tokenization/model settings
# Default to GPT-2 for easy Colab usage. Change both to Gemma if you have access.
TOKENIZER_ID = os.environ.get("LINEBREAK_TOKENIZER", "gpt2")
MAX_TOKENS = 2048 if TOKENIZER_ID == "gpt2" else 8192

MODEL_ID = os.environ.get("LINEBREAK_MODEL", "gpt2-small")  # set to "google/gemma-2-27b" if licensed + huge GPU

OUTPUT_DIR = "linebreak_data"
RESULTS_DIR = "results"

# Device / dtype
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DTYPE = torch.bfloat16 if DEVICE in ("cuda",) else torch.float32

console.print(f"[cyan]Device:[/cyan] {DEVICE}, dtype={DTYPE}")

# %%
# Utilities (subset from linebreak_utils.py + additions used by both flows)

def download_gutenberg_book(book_id: int) -> str:
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        alt = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
        r = requests.get(alt, timeout=30)
        r.raise_for_status()
        return r.text


def clean_gutenberg_text(text: str) -> str:
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    for m in start_markers:
        if m in text:
            text = text.split(m, 1)[1]
            break
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
    ]
    for m in end_markers:
        if m in text:
            text = text.split(m, 1)[0]
            break
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("***"):
        lines = lines[1:]
    text = "\n".join(lines)

    text = re.sub(r"\[Illustration:.*?\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[Footnote.*?:.*?\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[Illustration\]", "", text)
    text = re.sub(r"\[Footnote \d+\]", "", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    paras = text.split("\n\n")
    keep = []
    for p in paras:
        p = " ".join(p.split()).strip()
        if len(p) <= 100:
            continue
        if p.isupper():
            continue
        if p.startswith("CHAPTER ") or p.startswith("Chapter "):
            continue
        if p.startswith("CONTENTS") or p.startswith("Contents"):
            continue
        special_ratio = sum(1 for c in p if c in "_*[]()") / len(p)
        if special_ratio > 0.1:
            continue
        low = p.lower()
        fm = [
            "preface", "introduction", "frontispiece", "title-page", "dedication",
            "table of contents", "list of illustrations", "copyright", "publisher",
        ]
        if any(k in low[:50] for k in fm):
            if not ("\"" in p or "\"" in p or "\"" in p):
                continue
        keep.append(p)
    return "\n\n".join(keep)


def wrap_text_to_width(text: str, width: int) -> str:
    paras = text.split("\n\n")
    wrapped_paras = []
    for para in paras:
        para = para.replace("\n", " ")
        para = re.sub(r" +", " ", para).strip()
        if not para:
            continue
        words = para.split(" ")
        cur, cur_len = [], 0
        lines = []
        for w in words:
            wlen = len(w)
            if cur and cur_len + 1 + wlen > width:
                lines.append(" ".join(cur))
                cur = [w]; cur_len = wlen
            else:
                cur_len = (cur_len + 1 + wlen) if cur else wlen
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
        wrapped_paras.append("\n".join(lines))
    return "\n\n".join(wrapped_paras)


def validate_wrapped_text(text: str, width: int) -> bool:
    return all(len(line) <= width for line in text.split("\n"))


def paragraph_has_word_exceeding_width(text: str, width: int) -> bool:
    if width <= 0:
        return True
    norm = text.replace("\n", " ")
    norm = re.sub(r" +", " ", norm).strip()
    if not norm:
        return False
    for tok in re.split(r"\s+", norm):
        if len(tok) > width:
            return True
    return False


def get_hf_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def compute_line_features_per_token(text: str, token_ids: List[int], tokenizer: AutoTokenizer, max_length: Optional[int] = None):
    """
    Compute per-token features using offsets when available; else fallback via decode.
    - char_count: characters since the most recent newline after consuming the token
    - contains_newline: 1 if the token's span contains '\n', else 0
    """
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            max_length=(max_length if max_length else len(token_ids)),
            truncation=True,
        )
        input_ids = enc["input_ids"][0] if isinstance(enc["input_ids"][0], list) else enc["input_ids"]
        offsets = enc["offset_mapping"][0] if isinstance(enc["offset_mapping"][0], list) else enc["offset_mapping"]
        if len(input_ids) != len(token_ids):
            raise ValueError("Tokenizer mismatch with provided token_ids")

        n = len(text)
        last_nl = [-1] * (n + 1)
        last = -1
        for i, ch in enumerate(text):
            if ch == "\n":
                last = i
            last_nl[i + 1] = last

        char_counts, contains_newlines = [], []
        for (start, end) in offsets:
            if start == end == 0 and char_counts:
                char_counts.append(char_counts[-1])
                contains_newlines.append(0)
                continue
            span = text[start:end]
            contains_newlines.append(1 if "\n" in span else 0)
            ln_idx = last_nl[end]
            count = end - (ln_idx + 1)
            if count < 0:
                count = 0
            char_counts.append(count)
        return char_counts, contains_newlines
    except Exception:
        char_counts, contains_newlines = [], []
        line_count, at_line_start = 0, True
        for tok in token_ids:
            piece = tokenizer.decode([tok]).replace("\r\n", "\n").replace("\r", "\n")
            contains_newlines.append(1 if "\n" in piece else 0)
            for ch in piece:
                if ch == "\n":
                    line_count = 0
                    at_line_start = True
                else:
                    if at_line_start and ch == " ":
                        continue
                    at_line_start = False
                    line_count += 1
            char_counts.append(line_count)
        return char_counts, contains_newlines


# Streaming datasets (from gpt2_small_cell.py)
class FixedContextStream(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str | Path, widths: int | Sequence[int], context_length: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.widths = [int(widths)] if isinstance(widths, int) else [int(x) for x in widths]
        self.L = int(context_length)

    def _iter_width(self, w: int) -> Iterator[Dict[str, torch.Tensor]]:
        t_file = self.data_dir / f"linebreak_width_{w}_tokens.pt"
        c_file = self.data_dir / f"linebreak_width_{w}_char_count.pt"
        n_file = self.data_dir / f"linebreak_width_{w}_contains_newline.pt"
        assert t_file.exists() and c_file.exists() and n_file.exists(), f"Missing files for width {w}"
        t = torch.load(t_file, weights_only=False)
        c = torch.load(c_file, weights_only=False)
        n = torch.load(n_file, weights_only=False)

        json_file = self.data_dir / f"linebreak_width_{w}_tokens.json"
        seq_lengths = None
        if json_file.exists():
            try:
                import json as _json
                seqs = _json.loads(json_file.read_text())
                seq_lengths = [len(s) for s in seqs]
            except Exception:
                seq_lengths = None

        for i in range(t.shape[0]):
            if seq_lengths is None:
                row = t[i].tolist(); j = len(row)
                while j > 0 and row[j - 1] == 0:
                    j -= 1
                true_len = max(0, j)
            else:
                true_len = int(seq_lengths[i])
            if true_len >= self.L:
                yield {
                    "tokens": t[i, : self.L],
                    "char_count": c[i, : self.L],
                    "contains_newline": n[i, : self.L],
                    "width": torch.tensor(w, dtype=torch.long),
                }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for w in self.widths:
            yield from self._iter_width(int(w))


class RandomWidthStream(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str | Path, widths: int | Sequence[int], context_length: int, seed: int | None = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.widths = [int(widths)] if isinstance(widths, int) else [int(x) for x in widths]
        self.L = int(context_length)
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        active = []
        for w in self.widths:
            t_file = self.data_dir / f"linebreak_width_{w}_tokens.pt"
            c_file = self.data_dir / f"linebreak_width_{w}_char_count.pt"
            n_file = self.data_dir / f"linebreak_width_{w}_contains_newline.pt"
            if not (t_file.exists() and c_file.exists() and n_file.exists()):
                continue
            t = torch.load(t_file, weights_only=False)
            c = torch.load(c_file, weights_only=False)
            n = torch.load(n_file, weights_only=False)
            json_file = self.data_dir / f"linebreak_width_{w}_tokens.json"
            seq_lengths = None
            if json_file.exists():
                try:
                    import json as _json
                    seqs = _json.loads(json_file.read_text())
                    seq_lengths = [len(s) for s in seqs]
                except Exception:
                    seq_lengths = None
            valid = []
            for i in range(t.shape[0]):
                if seq_lengths is None:
                    row = t[i].tolist(); j = len(row)
                    while j > 0 and row[j - 1] == 0:
                        j -= 1
                    L = j
                else:
                    L = int(seq_lengths[i])
                if L >= self.L:
                    valid.append(i)
            if not valid:
                continue
            rng.shuffle(valid)
            active.append({"w": w, "t": t, "c": c, "n": n, "idxs": valid, "pos": 0})

        while active:
            k = rng.randrange(len(active))
            slot = active[k]
            pos = slot["pos"]
            if pos >= len(slot["idxs"]):
                active.pop(k)
                continue
            i = slot["idxs"][pos]
            slot["pos"] += 1
            yield {
                "tokens": slot["t"][i, : self.L],
                "char_count": slot["c"][i, : self.L],
                "contains_newline": slot["n"][i, : self.L],
                "width": torch.tensor(slot["w"], dtype=torch.long),
            }


# %% [markdown]
# Explore a single book and prepare dataset

# %%
# Download and clean a single book to preview wrapping
console.print("\n[bold cyan]═══ EXPLORING A SINGLE BOOK ═══[/bold cyan]\n")
book_id = BOOK_IDS[0]
console.print(f"Downloading book {book_id}...")
text = download_gutenberg_book(book_id)
console.print(f"[green]✓[/green] Downloaded: {len(text):,} characters")

console.print("Cleaning text...")
clean_text = clean_gutenberg_text(text)
paragraphs = [p for p in clean_text.split("\n\n") if len(p) >= MIN_SEQ_LENGTH]
console.print(f"[green]✓[/green] Cleaned: {len(clean_text):,} chars; {len(paragraphs)} substantial paragraphs")

sample_para = paragraphs[5] if len(paragraphs) > 5 else paragraphs[0]
sample_para = sample_para[:500]
console.print("\n[yellow]Original paragraph (first 500 chars):[/yellow]")
console.print(sample_para + "\n")

for w in [40, 80, 120]:
    wrapped = wrap_text_to_width(sample_para, w)
    console.print(f"[bold]Width {w}:[/bold]")
    console.print(wrapped)
    console.print()


# %%
# Create full dataset across widths
console.print("\n[bold cyan]═══ CREATING FULL DATASET ═══[/bold cyan]\n")
with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
    task1 = progress.add_task("[cyan]Downloading books...", total=len(BOOK_IDS))
    all_books = []
    for bid in BOOK_IDS:
        t = download_gutenberg_book(bid)
        all_books.append(clean_gutenberg_text(t))
        progress.update(task1, advance=1)

    console.print("\n[cyan]Extracting paragraphs...[/cyan]")
    all_paragraphs = []
    for bt in all_books:
        all_paragraphs.extend([p for p in bt.split("\n\n") if len(p) >= MIN_SEQ_LENGTH])

    task2 = progress.add_task("[cyan]Creating wrapped sequences...", total=len(LINE_WIDTHS))
    dataset: Dict[int, List[str]] = {}
    for width in LINE_WIDTHS:
        seqs = []
        for para in all_paragraphs:
            if paragraph_has_word_exceeding_width(para, width):
                continue
            wrapped = wrap_text_to_width(para, width)
            assert validate_wrapped_text(wrapped, width)
            seqs.append(wrapped)
            if len(seqs) >= NUM_SEQUENCES:
                break
        dataset[width] = seqs
        progress.update(task2, advance=1)

console.print(f"[green]✓[/green] Created dataset with {len(dataset)} widths; total sequences: {sum(len(v) for v in dataset.values())}")


# %%
# Dataset statistics
console.print("\n[bold cyan]═══ DATASET STATISTICS ═══[/bold cyan]")
sample_widths = [15, 40, 80, 120, 150]
console.print("[yellow]Per-width statistics:[/yellow]")
for w in sample_widths:
    if w in dataset:
        seqs = dataset[w]
        if not seqs:
            continue
        avg_len = sum(len(s) for s in seqs) / len(seqs)
        console.print(f"  Width {w:3d}: {len(seqs)} sequences, avg {avg_len:,.0f} chars")

first_w = sorted(dataset.keys())[0]
all_text = "\n\n".join(dataset[first_w]) if dataset[first_w] else ""
console.print("\n[yellow]Overall statistics:[/yellow]")
console.print(f"  Total unique sequences: {len(dataset[first_w])}")
console.print(f"  Total characters (unformatted): {len(all_text):,}")
console.print(f"  Unique characters: {len(set(all_text)) if all_text else 0}")

cnt = Counter(all_text)
console.print("\n[yellow]Most common characters:[/yellow]")
for ch, c in cnt.most_common(10):
    disp = repr(ch) if ch in "\n\t " else ch
    pct = (100 * c / len(all_text)) if all_text else 0
    console.print(f"  {disp}: {c:,} ({pct:.2f}%)")


# %%
# Save raw text per width
console.print("\n[bold cyan]═══ SAVING TEXT FILES ═══[/bold cyan]")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with Progress(SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
    task = progress.add_task("[cyan]Saving text files...", total=len(LINE_WIDTHS))
    for width, seqs in dataset.items():
        out = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n\n=====\n\n".join(seqs))
        progress.update(task, advance=1)
console.print(f"[green]✓[/green] Saved {len(dataset)} text files to '{OUTPUT_DIR}/'")


# %%
# Tokenization using a configurable tokenizer
console.print("\n[bold cyan]═══ TOKENIZING ═══[/bold cyan]")
console.print(f"[yellow]Loading tokenizer:[/yellow] {TOKENIZER_ID}")
tokenizer = get_hf_tokenizer(TOKENIZER_ID)
console.print("[green]✓[/green] Tokenizer loaded")

tokenized_dataset: Dict[int, List[List[int]]] = {}
with Progress(SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
    task = progress.add_task("[cyan]Tokenizing sequences...", total=len(LINE_WIDTHS))
    for width in LINE_WIDTHS:
        seqs = dataset[width]
        toks = []
        for s in seqs:
            ids = tokenizer.encode(s, max_length=MAX_TOKENS, truncation=True)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            toks.append(ids)
        tokenized_dataset[width] = toks
        progress.update(task, advance=1)
console.print(f"[green]✓[/green] Tokenized {len(tokenized_dataset)} widths")

# Show sample tokens
if 40 in tokenized_dataset and tokenized_dataset[40]:
    sample_tokens = tokenized_dataset[40][0][:100]
    console.print(f"\n[yellow]Sample tokens (width 40, first 100):[/yellow]")
    console.print(f"  {len(sample_tokens)} tokens: {sample_tokens[:20]}...")
    console.print(f"  Decoded: {tokenizer.decode(sample_tokens)[:150]}...")


# %%
# Compute per-token line features
console.print("\n[bold cyan]═══ COMPUTING LINE FEATURES ═══[/bold cyan]")
char_count_dataset: Dict[int, List[List[int]]] = {}
contains_newline_dataset: Dict[int, List[List[int]]] = {}
with Progress(SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
    task = progress.add_task("[cyan]Computing line features...", total=len(LINE_WIDTHS))
    for width in LINE_WIDTHS:
        texts = dataset[width]
        seqs = tokenized_dataset[width]
        cc_seqs, nl_seqs = [], []
        for text, ids in zip(texts, seqs):
            cc, nl = compute_line_features_per_token(text, ids, tokenizer, max_length=MAX_TOKENS)
            cc_seqs.append(cc)
            nl_seqs.append(nl)
        char_count_dataset[width] = cc_seqs
        contains_newline_dataset[width] = nl_seqs
        progress.update(task, advance=1)
console.print(f"[green]✓[/green] Computed line features for {len(char_count_dataset)} widths")

if 40 in char_count_dataset and char_count_dataset[40]:
    console.print("\n[yellow]Sample features (width 40, first 10 tokens):[/yellow]")
    console.print(f"  char_count: {char_count_dataset[40][0][:10]}")
    console.print(f"  contains_newline: {contains_newline_dataset[40][0][:10]}")


# %%
# Save tokenized data and features
console.print("\n[bold cyan]═══ SAVING TOKENIZED DATA & FEATURES ═══[/bold cyan]")
with Progress(SpinnerColumn(), *Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
    task = progress.add_task("[cyan]Saving...", total=len(LINE_WIDTHS))
    for width in LINE_WIDTHS:
        # JSON tokens
        json_file = os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_tokens.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(tokenized_dataset[width], f)

        # PyTorch tensors (padded to max len per width)
        max_len = max((len(seq) for seq in tokenized_dataset[width]), default=0)
        padded = [seq + [0] * (max_len - len(seq)) for seq in tokenized_dataset[width]]
        tokens_tensor = torch.tensor(padded, dtype=torch.long)
        torch.save(tokens_tensor, os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_tokens.pt"))

        char_padded = [seq + [0] * (max_len - len(seq)) for seq in char_count_dataset[width]]
        torch.save(torch.tensor(char_padded, dtype=torch.long), os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_char_count.pt"))

        nl_padded = [seq + [0] * (max_len - len(seq)) for seq in contains_newline_dataset[width]]
        torch.save(torch.tensor(nl_padded, dtype=torch.long), os.path.join(OUTPUT_DIR, f"linebreak_width_{width}_contains_newline.pt"))

        progress.update(task, advance=1)

console.print(f"\n[bold green]✓ DATASET COMPLETE![/bold green]")
console.print(f"Saved to '[cyan]{OUTPUT_DIR}/[/cyan]'.")
console.print("  • linebreak_width_*.txt (raw text)")
console.print("  • linebreak_width_*_tokens.json (token IDs as JSON)")
console.print("  • linebreak_width_*_tokens.pt (token tensors)")
console.print("  • linebreak_width_*_char_count.pt (chars since last newline per token)")
console.print("  • linebreak_width_*_contains_newline.pt (1 if token contains \\n)")


# %% [markdown]
# Model: load with TransformerLens and stream batches

# %%
console.print("\n[bold cyan]═══ LOADING MODEL ═══[/bold cyan]")
try:
    model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=DTYPE)
    console.print(f"[green]✓[/green] Loaded model: {MODEL_ID}")
except Exception as e:
    console.print(f"[red]✗ Failed to load '{MODEL_ID}'[/red]: {e}")
    console.print("[yellow]Falling back to 'gpt2-small'[/yellow]")
    MODEL_ID = "gpt2-small"
    model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=DTYPE)
    console.print(f"[green]✓[/green] Loaded fallback: {MODEL_ID}")

CTX = model.cfg.n_ctx
STREAM_WIDTHS = list(range(15, 151, 5))

# Choose stream type
stream_ds = RandomWidthStream(OUTPUT_DIR, STREAM_WIDTHS, CTX, seed=42)
stream_loader = torch.utils.data.DataLoader(stream_ds, batch_size=32)

console.print(f"Streaming datasets at context length L={CTX}")


# %%
# Aggregate means by char_count (skip tokens whose span contains newline)
LAYER = 1
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
MIN_CHAR = 31
MAX_CHAR = 119
MAX_BATCHES = 10

class OnlineMean:
    def __init__(self, dim: int):
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float32)
    def update(self, x: torch.Tensor):
        if x.device.type != "cpu":
            x = x.to(device="cpu")
        x = x.to(dtype=torch.float32)
        self.n += 1
        if self.n == 1:
            self.mean.copy_(x)
        else:
            self.mean.add_((x - self.mean) / self.n)

stats: Dict[int, OnlineMean] = {}

with Progress(SpinnerColumn(), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", MofNCompleteColumn(), TimeElapsedColumn(), TextColumn("{task.description}"), transient=True, console=console) as progress:
    task = progress.add_task("[cyan]Processing batches", total=MAX_BATCHES)
    for i, batch in enumerate(stream_loader):
        tokens = batch["tokens"].to(model.cfg.device)
        char_count = batch["char_count"]  # CPU
        contains_nl = batch["contains_newline"]  # CPU
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME, stop_at_layer=LAYER + 1, return_type="logits")
        acts = cache[HOOK_NAME]  # [B, T, d_model]
        d_model = int(acts.shape[-1])
        for b in range(acts.shape[0]):
            for t in range(acts.shape[1]):
                if int(contains_nl[b, t]) != 0:
                    continue
                k = int(char_count[b, t])
                if k < MIN_CHAR or k > MAX_CHAR:
                    continue
                if k not in stats:
                    stats[k] = OnlineMean(d_model)
                stats[k].update(acts[b, t])
        progress.update(task, advance=1)
        if (i + 1) >= MAX_BATCHES:
            break

charcount_means = {k: s.mean.clone() for k, s in stats.items() if s.n > 0}
console.print({
    "buckets": len(charcount_means),
    "example_keys": sorted(list(charcount_means.keys()))[:10],
    "vec_dim": (next(iter(charcount_means.values())).shape[0] if charcount_means else 0),
})


# %%
# PCA (3D) and Plotly visualization
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

if not charcount_means:
    raise RuntimeError("charcount_means not found. Run the aggregation cell first.")

_minc, _maxc = MIN_CHAR, MAX_CHAR
keys = sorted(k for k in charcount_means.keys() if (_minc <= k <= _maxc))
X = np.stack([charcount_means[k].numpy() for k in keys], axis=0) if keys else np.zeros((0, 3), dtype=np.float32)

pca = PCA(n_components=3, svd_solver="auto", random_state=0)
Z = pca.fit_transform(X) if len(X) >= 3 else np.zeros((len(X), 3), dtype=np.float32)
evr = pca.explained_variance_ratio_ if len(X) >= 3 else [0.0, 0.0, 0.0]

fig = go.Figure()

dark = "#141413"; light = "#faf9f5"; mid = "#b0aea5"; light_gray = "#e8e6dc"
orange = "#d97757"; blue = "#6a9bcc"; green = "#788c5d"
brand_scale = [[0.0, orange], [0.5, blue], [1.0, green]]

_cmin = _minc if _minc is not None else (keys[0] if keys else 0)
_cmax = _maxc if _maxc is not None else (keys[-1] if keys else 1)

if len(keys):
    fig.add_trace(
        go.Scatter3d(
            x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=keys,
                colorscale=brand_scale,
                cmin=_cmin,
                cmax=_cmax,
                showscale=True,
                colorbar=dict(title=dict(text="char_count", font=dict(family="Arial", color=dark)), bgcolor=light),
                line=dict(color=light, width=0.6),
            ),
            text=[f"k={k}" for k in keys],
            hovertemplate=("<b>%{text}</b><br>" + "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>"),
            name="char_count means",
        )
    )

# Helper for brand color gradient
def _hex_to_rgb(h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0,2,4))
def _rgb_to_hex(rgb): return '#%02x%02x%02x' % rgb
def _lerp(a,b,t): return tuple(int(round(a[i] + (b[i]-a[i])*t)) for i in range(3))
_o,_b,_g = _hex_to_rgb(orange), _hex_to_rgb(blue), _hex_to_rgb(green)
def brand_color_for_k(k, kmin, kmax):
    t = 0.0 if kmax==kmin else (k - kmin)/(kmax - kmin)
    c = _lerp(_o, _b, t/0.5) if t <= 0.5 else _lerp(_b, _g, (t-0.5)/0.5)
    return _rgb_to_hex(c)

if len(keys) >= 2:
    kmin, kmax = (_cmin, _cmax)
    for i in range(len(keys)-1):
        k_mid = 0.5*(keys[i] + keys[i+1])
        c = brand_color_for_k(k_mid, kmin, kmax)
        fig.add_trace(go.Scatter3d(x=[Z[i,0], Z[i+1,0]], y=[Z[i,1], Z[i+1,1]], z=[Z[i,2], Z[i+1,2]], mode='lines', line=dict(color=c, width=3), hoverinfo='skip', showlegend=False))

fig.update_layout(
    title=dict(text=f"PCA of Char Count Means — EVR: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f}", x=0.02),
    paper_bgcolor=light,
    scene=dict(
        xaxis=dict(title=dict(text="PC1"), gridcolor=light_gray, backgroundcolor=light),
        yaxis=dict(title=dict(text="PC2"), gridcolor=light_gray, backgroundcolor=light),
        zaxis=dict(title=dict(text="PC3"), gridcolor=light_gray, backgroundcolor=light),
    ),
    margin=dict(l=10, r=10, t=50, b=10), height=640,
)

try:
    fig.show()
except Exception:
    import plotly.io as pio
    pio.renderers.default = "browser"
    try:
        fig.show()
    except Exception:
        pass

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
out_file = os.path.join(RESULTS_DIR, "charcount_pca_3d_pc1_3.html")
fig.write_html(out_file, include_plotlyjs="cdn")
console.print({"saved_pc1_3": out_file})

