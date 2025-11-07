"""
Utilities for creating linebreaking datasets from Project Gutenberg texts.

This module implements the dataset curation methodology from "When Models Manipulate 
Manifolds" (Anthropic), where text is wrapped to fixed character widths at word 
boundaries to study model behavior on synthetic formatting variations.
"""

import re
import requests
import json
import torch
from typing import List, Dict, Optional
from pathlib import Path
from transformer_lens import HookedTransformer


def download_gutenberg_book(book_id: int) -> str:
    """
    Download a book from Project Gutenberg by its ID.
    
    Args:
        book_id: The Project Gutenberg book ID (e.g., 1342 for Pride & Prejudice)
    
    Returns:
        The full text of the book as a string
        
    Raises:
        requests.RequestException: If the download fails
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        # Try alternative URL format
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text


def clean_gutenberg_text(text: str) -> str:
    """
    Clean Project Gutenberg text by removing headers, footers, and normalizing whitespace.
    
    Project Gutenberg books contain licensing information at the start and end.
    This function strips those sections and keeps only substantial paragraphs.
    
    Args:
        text: Raw text from Project Gutenberg
        
    Returns:
        Cleaned text with only the book content, substantial paragraphs preserved
    """
    # Remove Project Gutenberg header (everything before "*** START OF")
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT"
    ]
    
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    
    # Remove Project Gutenberg footer (everything after "*** END OF")
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG"
    ]
    
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    
    # Remove the remaining header line (usually just "***" or the ebook title)
    lines = text.split('\n')
    if lines and lines[0].strip().startswith('***'):
        lines = lines[1:]
    
    # Rejoin and normalize whitespace
    text = '\n'.join(lines)
    
    # Remove common Project Gutenberg artifacts
    # Remove [Illustration: ...] markers
    text = re.sub(r'\[Illustration:.*?\]', '', text, flags=re.DOTALL)
    # Remove [Footnote: ...] markers
    text = re.sub(r'\[Footnote.*?:.*?\]', '', text, flags=re.DOTALL)
    # Remove standalone illustration/footnote references
    text = re.sub(r'\[Illustration\]', '', text)
    text = re.sub(r'\[Footnote \d+\]', '', text)
    
    # Replace multiple blank lines with double newline (paragraph separator)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Split into paragraphs (separated by blank lines)
    paragraphs = text.split('\n\n')
    
    # Process each paragraph
    substantial_paragraphs = []
    for para in paragraphs:
        # Join lines within paragraph, normalize spaces
        para = ' '.join(para.split())
        para = para.strip()
        
        # Skip paragraphs that are just artifacts or too short
        if len(para) <= 100:
            continue
        
        # Skip paragraphs that look like titles, headers, or page markers
        if para.isupper():
            continue
        if para.startswith('CHAPTER ') or para.startswith('Chapter '):
            continue
        if para.startswith('CONTENTS') or para.startswith('Contents'):
            continue
        
        # Skip paragraphs with too many special characters (likely artifacts)
        special_char_ratio = sum(1 for c in para if c in '_*[]()') / len(para)
        if special_char_ratio > 0.1:
            continue
        
        # Skip front matter patterns
        lower_para = para.lower()
        front_matter_keywords = ['preface', 'introduction', 'frontispiece', 
                                 'title-page', 'dedication', 'table of contents',
                                 'list of illustrations', 'copyright', 'publisher']
        if any(keyword in lower_para[:50] for keyword in front_matter_keywords):
            # But allow it if it's clearly narrative text (has quotes and proper sentences)
            if not ('"' in para or '"' in para or '"' in para):
                continue
        
        substantial_paragraphs.append(para)
    
    # Join paragraphs with double newline
    return '\n\n'.join(substantial_paragraphs)


def wrap_text_to_width(text: str, width: int) -> str:
    """
    Wrap text to a fixed character width, breaking at word boundaries.
    
    This implements the paper's methodology: insert newlines every k characters
    at the nearest word boundary ≤k. Never breaks in the middle of a word.
    
    Args:
        text: Input text (can contain paragraphs separated by double newlines)
        width: Maximum line width in characters
        
    Returns:
        Text wrapped to the specified width
    """
    paragraphs = text.split('\n\n')
    wrapped_paragraphs = []
    
    for para in paragraphs:
        # Remove existing line breaks within the paragraph
        para = para.replace('\n', ' ')
        # Normalize multiple spaces
        para = re.sub(r' +', ' ', para).strip()
        
        if not para:
            continue
            
        wrapped_lines = []
        words = para.split(' ')
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # If adding this word would exceed width
            if current_line and current_length + 1 + word_length > width:
                # Save current line and start a new one
                wrapped_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                # Add word to current line
                if current_line:
                    current_length += 1 + word_length  # +1 for space
                else:
                    current_length = word_length
                current_line.append(word)
        
        # Add the last line if it exists
        if current_line:
            wrapped_lines.append(' '.join(current_line))
        
        wrapped_paragraphs.append('\n'.join(wrapped_lines))
    
    return '\n\n'.join(wrapped_paragraphs)


def create_linebreak_dataset(
    book_ids: List[int], 
    num_sequences: int, 
    line_widths: List[int],
    min_seq_length: int = 500
) -> Dict[int, List[str]]:
    """
    Create a linebreaking dataset from multiple books at various widths.
    
    Downloads books from Project Gutenberg, cleans them, and creates sequences
    of wrapped text at different line widths. The same sequences are formatted
    at each width for controlled comparison.
    
    Sequences can be longer than your model's context window - just truncate
    to the desired length during training/evaluation (e.g., first 1024 tokens).
    
    Args:
        book_ids: List of Project Gutenberg book IDs to download
        num_sequences: Approximate number of text sequences to generate
        line_widths: List of line widths to generate (e.g., [40, 80, 120])
        min_seq_length: Minimum sequence length in characters (default: 500)
        
    Returns:
        Dictionary mapping line_width -> list of wrapped text sequences
    """
    print(f"Downloading and cleaning {len(book_ids)} books...")
    
    # Download and clean all books
    all_text = []
    for book_id in book_ids:
        try:
            print(f"  Downloading book {book_id}...", end=' ')
            text = download_gutenberg_book(book_id)
            cleaned = clean_gutenberg_text(text)
            all_text.append(cleaned)
            print(f"✓ ({len(cleaned):,} chars)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Combine all text
    combined_text = '\n\n'.join(all_text)
    
    # Split into paragraphs
    paragraphs = combined_text.split('\n\n')
    print(f"\nTotal paragraphs: {len(paragraphs)}")
    
    # Create sequences by grouping consecutive paragraphs
    # Group into sequences of approximately equal size
    sequences = []
    
    # Calculate approximate paragraphs per sequence
    paras_per_sequence = max(1, len(paragraphs) // num_sequences)
    
    for i in range(0, len(paragraphs), paras_per_sequence):
        chunk = paragraphs[i:i + paras_per_sequence]
        if chunk:
            seq_text = '\n\n'.join(chunk)
            # Only keep sequences that meet minimum length
            if len(seq_text) >= min_seq_length:
                sequences.append(seq_text)
    
    # If we have too many sequences, sample down to target number
    if len(sequences) > num_sequences:
        import random
        random.seed(42)  # For reproducibility
        sequences = random.sample(sequences, num_sequences)
    
    # Calculate actual length statistics
    if sequences:
        lengths = [len(s) for s in sequences]
        avg_len = sum(lengths) // len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"Created {len(sequences)} sequences (avg: {avg_len} chars, "
              f"range: {min_len}-{max_len})")
    else:
        print("Created 0 sequences")
    
    # Generate dataset for each width
    dataset = {}
    print(f"\nWrapping to {len(line_widths)} different widths...")
    for width in line_widths:
        print(f"  Width {width}...", end=' ')
        wrapped_sequences = [wrap_text_to_width(seq, width) for seq in sequences]
        dataset[width] = wrapped_sequences
        print("✓")
    
    return dataset


def get_gemma_tokenizer(device="mps", dtype="float16"):
    """
    Get the Gemma-2-2b tokenizer via TransformerLens.
    
    Args:
        device: Device to use ("mps" for Mac M3, "cuda" for NVIDIA, "cpu" for CPU)
        dtype: Data type ("float16" for faster/less memory, "float32" for precision)
    
    Returns:
        The Gemma-2-2b tokenizer from a HookedTransformer model
    """
    print("Loading Gemma-2-2b tokenizer via TransformerLens...")
    print(f"  Device: {device}, dtype: {dtype}")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        device=device,
        dtype=dtype
    )
    return model.tokenizer


def tokenize_dataset(
    dataset: Dict[int, List[str]], 
    tokenizer = None,
    max_length: Optional[int] = 8192,
    device: str = "mps",
    dtype: str = "float16"
) -> Dict[int, List[List[int]]]:
    """
    Tokenize all sequences in the dataset using Gemma-2-2b tokenizer.
    
    Args:
        dataset: Dictionary mapping line_width -> list of text sequences
        tokenizer: Gemma tokenizer (if None, will load it)
        max_length: Maximum sequence length in tokens (default: 8192 for Gemma-2-2b)
                   If None, no truncation is applied
        device: Device for model ("mps", "cuda", or "cpu")
        dtype: Data type ("float16" or "float32")
        
    Returns:
        Dictionary mapping line_width -> list of token sequences (as lists of ints)
    """
    if tokenizer is None:
        tokenizer = get_gemma_tokenizer(device=device, dtype=dtype)
    
    print(f"\nTokenizing dataset (max_length={max_length if max_length else 'None'})...")
    
    tokenized_dataset = {}
    for width in sorted(dataset.keys()):
        sequences = dataset[width]
        tokenized_sequences = []
        
        for seq in sequences:
            # Tokenize the sequence
            tokens = tokenizer.encode(seq)
            
            # Truncate if max_length is specified
            if max_length is not None and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            tokenized_sequences.append(tokens)
        
        tokenized_dataset[width] = tokenized_sequences
    
    # Print statistics
    first_width = sorted(tokenized_dataset.keys())[0]
    token_lengths = [len(seq) for seq in tokenized_dataset[first_width]]
    avg_len = sum(token_lengths) // len(token_lengths) if token_lengths else 0
    min_len = min(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    
    print(f"✓ Tokenized {len(tokenized_dataset)} widths")
    print(f"  Token lengths: avg={avg_len}, min={min_len}, max={max_len}")
    
    return tokenized_dataset


def generate_token_metadata(
    text: str,
    tokens: List[int],
    tokenizer,
    line_width: int
) -> List[Dict[str, int]]:
    """
    Generate metadata for each token position in a sequence.
    
    Tracks character position, line width, remaining space, token lengths, etc.
    This metadata helps the model understand where it is on each line and when
    to insert line breaks.
    
    Args:
        text: The original wrapped text
        tokens: List of token IDs
        tokenizer: Tokenizer to decode tokens
        line_width: The line width constraint (k)
        
    Returns:
        List of metadata dicts, one per token, with keys:
        - char_position: Current position in line (in characters)
        - line_width: The constraint k
        - chars_remaining: Space left on current line
        - token_length: Length of current token in characters
        - next_token_length: Length of next token in characters
        - line_number: Which line we're on (0-indexed)
        - is_newline: Whether this token contains/is a newline
    """
    metadata = []
    
    # Decode all tokens to get their string representations
    token_strings = []
    for token in tokens:
        decoded = tokenizer.decode([token])
        token_strings.append(decoded)
    
    # Track position as we go through tokens
    char_position = 0  # Position in current line
    line_number = 0
    
    for i, (token, token_str) in enumerate(zip(tokens, token_strings)):
        # Calculate token length
        token_length = len(token_str)
        
        # Calculate next token length
        next_token_length = len(token_strings[i + 1]) if i + 1 < len(token_strings) else 0
        
        # Check if this token contains a newline
        is_newline = '\n' in token_str
        
        # Calculate characters remaining on current line
        chars_remaining = line_width - char_position
        
        # Create metadata for this token
        token_metadata = {
            'char_position': char_position,
            'line_width': line_width,
            'chars_remaining': chars_remaining,
            'token_length': token_length,
            'next_token_length': next_token_length,
            'line_number': line_number,
            'is_newline': is_newline
        }
        metadata.append(token_metadata)
        
        # Update position for next token
        if is_newline:
            # Reset to start of new line
            # Handle case where token might be "\nword" - position is after newline
            parts = token_str.split('\n')
            if len(parts) > 1:
                # Token contains newline, position is length of text after last newline
                char_position = len(parts[-1])
                line_number += token_str.count('\n')
            else:
                char_position = 0
                line_number += 1
        else:
            # Continue on same line
            char_position += token_length
    
    return metadata


def create_metadata_dataset(
    dataset: Dict[int, List[str]],
    tokenized_dataset: Dict[int, List[List[int]]],
    tokenizer = None,
    device: str = "mps",
    dtype: str = "float16"
) -> Dict[int, List[List[Dict[str, int]]]]:
    """
    Generate metadata for all tokenized sequences in the dataset.
    
    Args:
        dataset: Dictionary mapping line_width -> list of text sequences
        tokenized_dataset: Dictionary mapping line_width -> list of token sequences
        tokenizer: Gemma tokenizer (if None, will load it)
        device: Device for model
        dtype: Data type
        
    Returns:
        Dictionary mapping line_width -> list of metadata sequences
        Each metadata sequence is a list of dicts with token position info
    """
    if tokenizer is None:
        tokenizer = get_gemma_tokenizer(device=device, dtype=dtype)
    
    print("\nGenerating token metadata...")
    
    metadata_dataset = {}
    for width in sorted(dataset.keys()):
        text_sequences = dataset[width]
        token_sequences = tokenized_dataset[width]
        
        metadata_sequences = []
        for text, tokens in zip(text_sequences, token_sequences):
            metadata = generate_token_metadata(text, tokens, tokenizer, width)
            metadata_sequences.append(metadata)
        
        metadata_dataset[width] = metadata_sequences
    
    print(f"✓ Generated metadata for {len(metadata_dataset)} widths")
    
    return metadata_dataset


def save_dataset(dataset: Dict[int, List[str]], output_dir: str) -> None:
    """
    Save the dataset to disk, with one file per line width.
    
    Each file contains all sequences for that width, separated by a delimiter.
    
    Args:
        dataset: Dictionary mapping line_width -> list of text sequences
        output_dir: Directory path where files will be saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving text dataset to {output_dir}/")
    
    for width, sequences in dataset.items():
        filename = output_path / f"linebreak_width_{width}.txt"
        
        # Join sequences with delimiter
        content = "\n\n=====\n\n".join(sequences)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = len(content) / 1024  # KB
        print(f"  width_{width}.txt: {len(sequences)} sequences, {file_size:.1f} KB")
    
    print(f"\n✓ Dataset saved successfully!")


def save_tokenized_dataset(
    tokenized_dataset: Dict[int, List[List[int]]], 
    output_dir: str
) -> None:
    """
    Save tokenized dataset to disk as JSON files.
    
    Each file contains token IDs for all sequences at that width.
    
    Args:
        tokenized_dataset: Dictionary mapping line_width -> list of token sequences
        output_dir: Directory path where files will be saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving tokenized dataset to {output_dir}/")
    
    for width, token_sequences in tokenized_dataset.items():
        filename = output_path / f"linebreak_width_{width}_tokens.json"
        
        # Save as JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(token_sequences, f)
        
        file_size = filename.stat().st_size / 1024  # KB
        total_tokens = sum(len(seq) for seq in token_sequences)
        print(f"  width_{width}_tokens.json: {len(token_sequences)} sequences, "
              f"{total_tokens:,} tokens, {file_size:.1f} KB")
    
    print(f"\n✓ Tokenized dataset saved successfully!")


def load_linebreak_batch(
    data_dir: str,
    widths: List[int],
    batch_size: int = 32,
    max_length: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load a batch of tokenized sequences and metadata from disk.
    
    Args:
        data_dir: Directory containing the .pt files (e.g., "linebreak_data")
        widths: List of line widths to load (e.g., [40, 80, 120])
        batch_size: Number of sequences to load per width
        max_length: Maximum sequence length (will truncate if provided)
        device: Device to load tensors to ("cpu", "mps", "cuda")
        
    Returns:
        Dictionary with keys:
        - 'tokens': Tensor of shape [total_sequences, seq_length]
        - 'metadata': Tensor of shape [total_sequences, seq_length, 7]
        - 'widths': Tensor of shape [total_sequences] indicating which width
        - 'width_values': List of actual width values for each sequence
    """
    data_path = Path(data_dir)
    
    all_tokens = []
    all_metadata = []
    all_width_labels = []
    all_width_values = []
    
    for width in widths:
        # Load tokens
        token_file = data_path / f"linebreak_width_{width}_tokens.pt"
        tokens = torch.load(token_file, map_location=device)
        
        # Load metadata
        metadata_file = data_path / f"linebreak_width_{width}_metadata.pt"
        metadata = torch.load(metadata_file, map_location=device)
        
        # Sample batch_size sequences
        num_sequences = tokens.shape[0]
        indices = torch.randperm(num_sequences)[:batch_size]
        
        batch_tokens = tokens[indices]
        batch_metadata = metadata[indices]
        
        # Truncate if needed
        if max_length is not None:
            batch_tokens = batch_tokens[:, :max_length]
            batch_metadata = batch_metadata[:, :max_length, :]
        
        all_tokens.append(batch_tokens)
        all_metadata.append(batch_metadata)
        
        # Track which width each sequence belongs to
        width_labels = torch.full((batch_size,), width, dtype=torch.long, device=device)
        all_width_labels.append(width_labels)
        all_width_values.extend([width] * batch_size)
    
    # Concatenate all batches
    tokens = torch.cat(all_tokens, dim=0)
    metadata = torch.cat(all_metadata, dim=0)
    width_labels = torch.cat(all_width_labels, dim=0)
    
    return {
        'tokens': tokens,
        'metadata': metadata,
        'widths': width_labels,
        'width_values': all_width_values
    }


def create_dataloader(
    data_dir: str,
    widths: List[int],
    batch_size: int = 32,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    device: str = "cpu"
):
    """
    Create a PyTorch-style dataloader for the linebreaking dataset.
    
    Args:
        data_dir: Directory containing the .pt files
        widths: List of line widths to include
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        device: Device to load tensors to
        
    Yields:
        Batches with keys: 'tokens', 'metadata', 'widths'
    """
    from torch.utils.data import Dataset, DataLoader
    
    class LineBreakDataset(Dataset):
        def __init__(self, data_dir, widths, max_length=None):
            self.data_path = Path(data_dir)
            self.widths = widths
            self.max_length = max_length
            
            # Load all data into memory
            self.all_tokens = []
            self.all_metadata = []
            self.all_width_labels = []
            
            for width in widths:
                # Load tokens
                token_file = self.data_path / f"linebreak_width_{width}_tokens.pt"
                tokens = torch.load(token_file)
                
                # Load metadata
                metadata_file = self.data_path / f"linebreak_width_{width}_metadata.pt"
                metadata = torch.load(metadata_file)
                
                # Truncate if needed
                if max_length is not None:
                    tokens = tokens[:, :max_length]
                    metadata = metadata[:, :max_length, :]
                
                self.all_tokens.append(tokens)
                self.all_metadata.append(metadata)
                
                # Track width labels
                num_sequences = tokens.shape[0]
                width_labels = torch.full((num_sequences,), width, dtype=torch.long)
                self.all_width_labels.append(width_labels)
            
            # Concatenate all
            self.tokens = torch.cat(self.all_tokens, dim=0)
            self.metadata = torch.cat(self.all_metadata, dim=0)
            self.widths = torch.cat(self.all_width_labels, dim=0)
        
        def __len__(self):
            return self.tokens.shape[0]
        
        def __getitem__(self, idx):
            return {
                'tokens': self.tokens[idx],
                'metadata': self.metadata[idx],
                'widths': self.widths[idx]
            }
    
    dataset = LineBreakDataset(data_dir, widths, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_tokenized_dataset_torch(
    tokenized_dataset: Dict[int, List[List[int]]], 
    output_dir: str,
    metadata_dataset: Optional[Dict[int, List[List[Dict[str, int]]]]] = None
) -> None:
    """
    Save tokenized dataset as PyTorch tensors (more efficient for training).
    
    Optionally includes metadata tensors for each token position.
    
    Args:
        tokenized_dataset: Dictionary mapping line_width -> list of token sequences
        output_dir: Directory path where files will be saved
        metadata_dataset: Optional metadata for each token position
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving tokenized dataset as PyTorch tensors to {output_dir}/")
    
    for width, token_sequences in tokenized_dataset.items():
        filename = output_path / f"linebreak_width_{width}_tokens.pt"
        
        # Convert to tensor (pad sequences to same length)
        max_len = max(len(seq) for seq in token_sequences)
        
        # Pad sequences
        padded_sequences = []
        for seq in token_sequences:
            padded = seq + [0] * (max_len - len(seq))  # Pad with 0
            padded_sequences.append(padded)
        
        # Convert to tensor
        tensor = torch.tensor(padded_sequences, dtype=torch.long)
        
        # Save tokens
        torch.save(tensor, filename)
        
        file_size = filename.stat().st_size / 1024  # KB
        print(f"  width_{width}_tokens.pt: shape {tensor.shape}, {file_size:.1f} KB")
        
        # Save metadata if provided
        if metadata_dataset and width in metadata_dataset:
            metadata_filename = output_path / f"linebreak_width_{width}_metadata.pt"
            metadata_sequences = metadata_dataset[width]
            
            # Convert metadata to tensors
            # Each metadata dict becomes a row of features
            metadata_keys = ['char_position', 'line_width', 'chars_remaining', 
                           'token_length', 'next_token_length', 'line_number', 'is_newline']
            
            padded_metadata = []
            for metadata_seq in metadata_sequences:
                # Convert each sequence's metadata to tensor
                seq_metadata = []
                for token_meta in metadata_seq:
                    features = [token_meta.get(key, 0) for key in metadata_keys]
                    seq_metadata.append(features)
                
                # Pad to max_len
                while len(seq_metadata) < max_len:
                    seq_metadata.append([0] * len(metadata_keys))
                
                padded_metadata.append(seq_metadata)
            
            # Convert to tensor: [num_sequences, max_len, num_features]
            metadata_tensor = torch.tensor(padded_metadata, dtype=torch.long)
            torch.save(metadata_tensor, metadata_filename)
            
            meta_size = metadata_filename.stat().st_size / 1024
            print(f"  width_{width}_metadata.pt: shape {metadata_tensor.shape}, {meta_size:.1f} KB")
    
    print(f"\n✓ Tokenized dataset saved as PyTorch tensors!")

