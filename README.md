# LineBreak Dataset Pipeline

Dataset curation for the linebreaking task from ["When Models Manipulate Manifolds"](https://arxiv.org/abs/2410.03380) (Anthropic).

Creates synthetic datasets where text is wrapped at different character widths to study how language models handle formatting variations.

---

## 🚀 Quick Start

```bash
# 1. Install
uv pip install -e .

# 2. Generate dataset (~10-15 min first time)
uv run python explore_dataset.py

# 3. Use it
python -c "from linebreak_utils import load_linebreak_batch; \
           batch = load_linebreak_batch('linebreak_data', [40], 8); \
           print(f'Loaded: {batch[\"tokens\"].shape}')"
```

**Output**: `linebreak_data/` with ~3.5GB of tokenized text at 28 different widths (15→150).

---

## 📖 What This Does

Recreates the paper's synthetic linebreaking dataset:

1. **Downloads** 10 classic books from Project Gutenberg
2. **Cleans** text (removes headers/footers, normalizes whitespace)
3. **Wraps** text at fixed widths (15, 20, 25, ..., 150 characters)
4. **Tokenizes** with Gemma-2-2b (8192 token context)
5. **Generates metadata** for each token:
   - `char_position`: Position in current line (0 to line_width)
   - `line_width`: The constraint k
   - `chars_remaining`: Space left on line
   - `token_length`: Current token size in characters
   - `next_token_length`: Next token size
   - `line_number`: Which line (0-indexed)
   - `is_newline`: Whether token contains newline

---

## 💻 Usage

### Load a Batch for Training

```python
from linebreak_utils import load_linebreak_batch

batch = load_linebreak_batch(
    data_dir="linebreak_data",
    widths=[40, 80, 120],       # Multiple widths
    batch_size=16,              # 16 per width = 48 total
    max_length=1024,            # Truncate to fit context
    device="mps"                # or "cuda" or "cpu"
)

tokens = batch['tokens']        # [48, 1024]
metadata = batch['metadata']    # [48, 1024, 7]
widths = batch['widths']        # [48]
```

### Use PyTorch DataLoader

```python
from linebreak_utils import create_dataloader

dataloader = create_dataloader(
    data_dir="linebreak_data",
    widths=[40, 80, 120],
    batch_size=32,
    max_length=1024,
    shuffle=True
)

for batch in dataloader:
    tokens = batch['tokens'].to(device)
    metadata = batch['metadata'].to(device)
    # ... train your model
```

### Generate Custom Dataset

```python
from linebreak_utils import (
    create_linebreak_dataset,
    get_gemma_tokenizer,
    tokenize_dataset,
    create_metadata_dataset,
    save_tokenized_dataset_torch
)

# Create custom dataset
dataset = create_linebreak_dataset(
    book_ids=[1342, 84],           # Just 2 books
    num_sequences=50,               # Fewer sequences
    line_widths=[40, 80],          # Just 2 widths
    min_seq_length=1000            # Longer sequences
)

# Tokenize
tokenizer = get_gemma_tokenizer()
tokenized = tokenize_dataset(dataset, tokenizer)
metadata = create_metadata_dataset(dataset, tokenized, tokenizer)

# Save
save_tokenized_dataset_torch(tokenized, "custom_data", metadata)
```

---

## 📊 Dataset Details

- **Books**: 10 classics (Pride & Prejudice, Alice in Wonderland, Sherlock Holmes, etc.)
- **Sequences**: ~200 per width (min 500 chars each)
- **Widths**: 28 different widths (15, 20, 25, ..., 150)
- **Tokenizer**: Gemma-2-2b (8192 token context, MPS/CUDA/CPU)
- **Total Size**: ~3.5GB (text + tokens + metadata)

### File Formats

Each width generates 3 files:

```python
linebreak_width_40.txt              # Raw text
linebreak_width_40_tokens.json      # Token IDs as JSON
linebreak_width_40_tokens.pt        # Tokens: [200, 8192]
linebreak_width_40_metadata.pt      # Metadata: [200, 8192, 7]
```

---

## 🔧 API Reference

### Core Functions

```python
# Dataset creation
create_linebreak_dataset(book_ids, num_sequences, line_widths, min_seq_length=500)

# Text processing
download_gutenberg_book(book_id: int) -> str
clean_gutenberg_text(text: str) -> str
wrap_text_to_width(text: str, width: int) -> str

# Tokenization
get_gemma_tokenizer(device="mps", dtype="float16")
tokenize_dataset(dataset, tokenizer, max_length=8192)
create_metadata_dataset(dataset, tokenized_dataset, tokenizer)

# Data loading
load_linebreak_batch(data_dir, widths, batch_size, max_length, device)
create_dataloader(data_dir, widths, batch_size, max_length, shuffle)

# Saving
save_dataset(dataset, output_dir)  # Save text files
save_tokenized_dataset(tokenized_dataset, output_dir)  # Save JSON
save_tokenized_dataset_torch(tokenized_dataset, output_dir, metadata_dataset)  # Save PyTorch
```

---

## 📁 Project Structure

```
LineBreakManifold/
├── linebreak_utils.py          # Core library
├── explore_dataset.py          # Pipeline script (9 cells)
├── examples/
│   ├── batch_loading_example.py
│   ├── metadata_example.py
│   └── load_example.py
├── pyproject.toml              # Dependencies
└── linebreak_data/             # Generated (gitignored)
    ├── linebreak_width_*.txt
    ├── linebreak_width_*_tokens.pt
    └── linebreak_width_*_metadata.pt
```

---

## 🛠️ Installation Options

```bash
# Basic (required only)
uv pip install -e .

# With dev tools (Jupyter, matplotlib)
uv pip install -e ".[dev]"

# With Scribe (interactive exploration)
uv pip install -e ".[scribe]"

# Everything
uv pip install -e ".[all]"
```

---

## 💡 Interactive Exploration (Optional)

Install [Scribe](https://github.com/goodfire-ai/scribe) for AI-assisted Jupyter notebooks:

```bash
uv pip install -e ".[scribe]"
scribe  # Launch interactive session
```

Then ask: *"Load a batch from width 40 and show me the metadata structure"*

All code and outputs auto-save to `notebooks/`.

---

## 🐛 Troubleshooting

### "No module named 'transformer_lens'"
```bash
uv pip install -e .
```

### "CUDA/MPS out of memory"
Reduce batch size or sequence length:
```python
batch = load_linebreak_batch(..., batch_size=8, max_length=1024)
```

### Slow first run
First run downloads Gemma-2-2b model (~5GB). This is cached.

### Dataset too large
Generate only the widths you need:
```python
LINE_WIDTHS = [40, 80, 120]  # Edit in explore_dataset.py
```

---

## 📚 Example Scripts

Check `examples/` for:
- `batch_loading_example.py` - How to load batches
- `metadata_example.py` - Understanding metadata
- `load_example.py` - Basic loading patterns

---

## 🎓 Citation

```bibtex
@article{anthropic2024manifolds,
  title={When Models Manipulate Manifolds},
  author={Anthropic},
  journal={arXiv preprint arXiv:2410.03380},
  year={2024}
}
```

---

## 📋 Requirements

- Python ≥3.11
- 10GB disk space
- 8GB RAM minimum
- GPU/MPS recommended (not required)

---

## 📝 License

MIT License - Free to use and modify for research.

---

**Questions?** Check the example scripts or ask in your group chat!
