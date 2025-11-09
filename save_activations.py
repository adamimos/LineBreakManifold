# %% Save Activations (width-by-width, shard per batch)
"""
Save per-token activations for the LineBreak dataset.

For each available width:
  - Load tokens (and optional char_count, contains_newline)
  - Run Gemma-2-27b and capture layer L residual stream activations
  - Save activations in per-batch shards to avoid OOM
  - Write a manifest with shard metadata

Output structure (default results/activations):
  results/
    activations/
      width_{W}/
        layer_{L}/
          manifest.json
          activations_{startIdx}_{endIdx}.pt    # {'activations_q': uint8[B,T,D], 'q_min': fp16[D], 'q_scale': fp16[D],
                                                #  'quantization': {'type': 'per_channel_uint8_minmax'}, 'start_idx', 'end_idx', 'width', 'layer'}
                                                # also includes 'char_count' and 'contains_newline' slices if available
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from rich.console import Console
from rich.progress import track
from transformer_lens import HookedTransformer


# ===== Defaults =====
DATA_DIR = Path("linebreak_data")
RESULTS_DIR = Path("results") / "activations"
LAYER = 1
BATCH_SIZE = 1
MAX_SEQUENCES: Optional[int] = None  # None to process all sequences


console = Console()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def discover_widths(data_dir: Path) -> List[int]:
    token_files = sorted(data_dir.glob("linebreak_width_*_tokens.pt"))
    widths = []
    for f in token_files:
        m = re.search(r"linebreak_width_(\d+)_tokens\.pt$", f.name)
        if m:
            widths.append(int(m.group(1)))
    return sorted(widths)


def load_model(device: str) -> HookedTransformer:
    console.print("[yellow]Loading Gemma-2-27b model...[/yellow]")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-27b",
        device=device,
        dtype=dtype
    )
    console.print(f"[green]✓ Model loaded[/green] (device={device})")
    return model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_width_io_paths(data_dir: Path, width: int) -> Dict[str, Path]:
    return {
        "tokens": data_dir / f"linebreak_width_{width}_tokens.pt",
        "char_count": data_dir / f"linebreak_width_{width}_char_count.pt",
        "contains_newline": data_dir / f"linebreak_width_{width}_contains_newline.pt",
    }


def get_width_out_dir(results_dir: Path, width: int, layer: int) -> Path:
    return results_dir / f"width_{width}" / f"layer_{layer}"


def save_manifest(out_dir: Path, manifest: Dict) -> None:
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def process_width(
    model: HookedTransformer,
    width: int,
    device: str,
    layer: int,
    batch_size: int,
    data_dir: Path,
    results_dir: Path,
    max_sequences: Optional[int] = None
) -> None:
    io_paths = get_width_io_paths(data_dir, width)
    if not io_paths["tokens"].exists():
        console.print(f"[yellow]⚠ Tokens not found for width {width}: {io_paths['tokens']}[/yellow]")
        return

    tokens = torch.load(io_paths["tokens"], weights_only=False)
    num_sequences, seq_len = tokens.shape[0], tokens.shape[1]

    # Optionally trim number of sequences
    if max_sequences is not None:
        num_sequences = min(num_sequences, max_sequences)
        tokens = tokens[:num_sequences]

    # Optional features
    char_count = None
    contains_newline = None
    if io_paths["char_count"].exists():
        char_count = torch.load(io_paths["char_count"], weights_only=False)[:num_sequences]
    if io_paths["contains_newline"].exists():
        contains_newline = torch.load(io_paths["contains_newline"], weights_only=False)[:num_sequences]

    out_dir = get_width_out_dir(results_dir, width, layer)
    ensure_dir(out_dir)

    # Initialize/overwrite manifest
    manifest = {
        "width": width,
        "layer": layer,
        "seq_len": int(seq_len),
        "num_sequences": int(num_sequences),
        "d_model": int(model.cfg.d_model),
        "batch_size": int(batch_size),
        "shards": []
    }
    save_manifest(out_dir, manifest)

    hook_name = f"blocks.{layer}.hook_resid_post"

    num_batches = (num_sequences + batch_size - 1) // batch_size
    console.print(f"[cyan]Width {width}[/cyan] • sequences={num_sequences}, seq_len={seq_len}, batches={num_batches}")

    for batch_idx in track(range(num_batches), description=f"W{width} L{layer}: saving activations"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_sequences)

        batch_tokens = tokens[start_idx:end_idx].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=hook_name,
                stop_at_layer=layer + 1
            )
        # Activations on CPU for quantization
        acts = cache[hook_name].cpu()  # [B, T, d_model]
        B, T, D = acts.shape
        X = acts.view(-1, D)  # [B*T, D]

        # Per-channel uint8 min/max quantization
        x_min = X.min(dim=0).values
        x_max = X.max(dim=0).values
        scale = (x_max - x_min).clamp(min=1e-12) / 255.0
        Q = torch.clamp(((X - x_min) / scale).round(), 0, 255).to(torch.uint8).view(B, T, D)

        shard: Dict[str, torch.Tensor] = {
            "activations_q": Q,                       # uint8 [B, T, D]
            "q_min": x_min.to(torch.float16),         # fp16 [D]
            "q_scale": scale.to(torch.float16),       # fp16 [D]
        }
        # Include aligned features if available
        if char_count is not None:
            shard["char_count"] = char_count[start_idx:end_idx].to(torch.int16)
        if contains_newline is not None:
            shard["contains_newline"] = contains_newline[start_idx:end_idx].to(torch.int8)

        # Write shard
        shard_file = out_dir / f"activations_{start_idx}_{end_idx}.pt"
        torch.save({
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "width": int(width),
            "layer": int(layer),
            "quantization": {"type": "per_channel_uint8_minmax"},
            **shard
        }, shard_file)

        # Update manifest incrementally
        manifest["shards"].append({
            "file": shard_file.name,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "shape": [B, T, D],
            "quantization": "per_channel_uint8_minmax"
        })
        save_manifest(out_dir, manifest)

        # Free memory
        del batch_tokens, cache, acts, X, Q, x_min, x_max, scale
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    console.print(f"[green]✓ Saved activations for width {width}[/green] → {out_dir}")


def main():
    device = detect_device()
    model = load_model(device)

    ensure_dir(RESULTS_DIR)

    widths = discover_widths(DATA_DIR)
    if not widths:
        console.print(f"[red]✗ No token files found in {DATA_DIR}[/red]")
        return

    console.print(f"[bold cyan]Saving activations[/bold cyan]")
    console.print(f"Layer: {LAYER}, Device: {device}, Batch size: {BATCH_SIZE}")
    console.print(f"Widths: {widths}")

    for width in widths:
        process_width(
            model=model,
            width=width,
            device=device,
            layer=LAYER,
            batch_size=BATCH_SIZE,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            max_sequences=MAX_SEQUENCES
        )

    console.print(f"[bold green]All activations saved![/bold green]")


if __name__ == "__main__":
    main()



# %%
