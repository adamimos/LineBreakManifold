# %% Cell 1: Setup
"""
Activation Analysis Pipeline for LineBreak Dataset
Collect layer 1 residual stream activations, aggregate by char_position, visualize with PCA
"""

import torch
import numpy as np
from pathlib import Path
from rich.progress import track
from rich.console import Console

# Setup
console = Console()
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# ===== Hyperparameters (single source of truth) =====
LAYER = 0
WIDTHS = np.arange(15, 151, 5)
BATCH_SIZE = 1         # sequences per batch
MAX_SEQUENCES = 5      # None for all sequences; small for testing
DATA_DIR = Path("linebreak_data")
RESULTS_DIR = Path("results")

# Ensure results dir exists
results_dir = RESULTS_DIR
results_dir.mkdir(exist_ok=True)

console.print(f"[green]✓ Setup complete[/green]")
console.print(f"Device: {device}")
console.print(f"Results directory: {results_dir}")
console.print(f"PyTorch version: {torch.__version__}")

# %% Cell 2: Load Model
"""
Load Gemma-2-27b with TransformerLens
"""

from transformer_lens import HookedTransformer

console.print("[yellow]Loading Gemma-2-27b model...[/yellow]")

dtype = torch.bfloat16 if device != "cpu" else torch.float32
model = HookedTransformer.from_pretrained(
    "google/gemma-2-27b",
    device=device,
    dtype=dtype
)

console.print(f"[green]✓ Model loaded[/green]")
console.print(f"Layers: {model.cfg.n_layers}")
console.print(f"d_model: {model.cfg.d_model}")
console.print(f"Model device: {next(model.parameters()).device}")

# %% Cell 3: Process All Widths
"""
Process all widths: load data in batches → collect layer 1 activations → aggregate by current char_position
Loops through all available widths automatically

Note: char_position is CURRENT position on line (before placing token), not after.
This resets to 0 after newlines and never exceeds the line width.
"""

from collections import defaultdict
import time

console.print(f"\n[bold cyan]Processing {len(WIDTHS)} widths for layer {LAYER}[/bold cyan]")
console.print(f"Widths: {WIDTHS}")

data_dir = DATA_DIR

for width_idx, TARGET_WIDTH in enumerate(WIDTHS, 1):
    console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
    console.print(f"[bold magenta]Width {TARGET_WIDTH} ({width_idx}/{len(WIDTHS)})[/bold magenta]")
    console.print(f"[bold magenta]{'='*60}[/bold magenta]")

    # Check if already processed
    output_file = results_dir / f"width_{TARGET_WIDTH}_layer{LAYER}_means.pt"
    if output_file.exists():
        console.print(f"[yellow]⚠ Already processed! Skipping: {output_file}[/yellow]")
        continue

    # Load data files directly
    tokens_file = data_dir / f"linebreak_width_{TARGET_WIDTH}_tokens.pt"
    metadata_file = data_dir / f"linebreak_width_{TARGET_WIDTH}_metadata.pt"

    if not tokens_file.exists():
        console.print(f"[yellow]⚠ Data file not found: {tokens_file}[/yellow]")
        console.print(f"[yellow]Skipping width {TARGET_WIDTH}[/yellow]")
        continue

    # Initialize aggregation structures
    position_sums = defaultdict(lambda: torch.zeros(model.cfg.d_model, dtype=torch.float32))
    position_counts = defaultdict(int)

    console.print(f"Loading data info from disk...")
    
    # Load just to check dimensions, then release
    temp_tokens = torch.load(tokens_file, weights_only=False)
    num_sequences = temp_tokens.shape[0]
    if MAX_SEQUENCES is not None:
        num_sequences = min(num_sequences, MAX_SEQUENCES)
    
    # Don't keep full dataset in memory - we'll load in chunks
    del temp_tokens
    
    num_batches = (num_sequences + BATCH_SIZE - 1) // BATCH_SIZE

    console.print(f"Total sequences to process: {num_sequences}")
    console.print(f"Processing in {num_batches} batches of {BATCH_SIZE}...")
    
    total_tokens = 0
    for batch_idx in track(range(num_batches), description=f"Processing width {TARGET_WIDTH}"):
        batch_start_time = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_sequences)

        # Load only this batch from disk (slower but avoids RAM issues)
        load_start = time.time()
        all_tokens = torch.load(tokens_file, weights_only=False)
        all_metadata = torch.load(metadata_file, weights_only=False)
        
        batch_tokens = all_tokens[start_idx:end_idx].to(device)
        batch_metadata = all_metadata[start_idx:end_idx]
        
        # Immediately free the full dataset
        del all_tokens, all_metadata
        load_time = time.time() - load_start

        # Run model and collect layer 1 activations
        model_start = time.time()
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=f"blocks.{LAYER}.hook_resid_post",
                stop_at_layer=LAYER+1
            )
        model_time = time.time() - model_start

        # Get activations: [batch, seq_len, d_model]
        activations = cache[f"blocks.{LAYER}.hook_resid_post"].cpu().float()

        # Extract analysis positions (new schema only):
        # Use after_pos and exclude newline tokens from analysis.
        after_pos = batch_metadata[:, :, 1]
        is_newline = batch_metadata[:, :, 7].bool()
        char_positions = after_pos.cpu().numpy()

        # Aggregate by char_position
        agg_start = time.time()
        batch_tokens_processed = 0
        for seq_idx in range(activations.shape[0]):
            for token_idx in range(activations.shape[1]):
                if is_newline[seq_idx, token_idx].item():
                    continue
                char_pos = int(char_positions[seq_idx, token_idx])
                position_sums[char_pos] += activations[seq_idx, token_idx]
                position_counts[char_pos] += 1
                total_tokens += 1
                batch_tokens_processed += 1
        agg_time = time.time() - agg_start

        # Free memory aggressively
        del batch_tokens, batch_metadata, cache, activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        batch_total_time = time.time() - batch_start_time
        
        # Print detailed timing for this batch (only every 10 batches to reduce noise)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            console.print(f"  [dim]Batch {batch_idx+1}/{num_batches}: "
                         f"Load={load_time:.1f}s | Model={model_time:.1f}s | "
                         f"Aggregate={agg_time:.1f}s | Total={batch_total_time:.1f}s | "
                         f"Tokens={batch_tokens_processed:,}[/dim]")

    # Compute means
    max_position = max(position_counts.keys())
    means = torch.zeros(max_position + 1, model.cfg.d_model)
    counts = torch.zeros(max_position + 1, dtype=torch.long)

    for pos in position_counts.keys():
        means[pos] = position_sums[pos] / position_counts[pos]
        counts[pos] = position_counts[pos]

    # Save results
    torch.save({
        'means': means,
        'counts': counts,
        'width': TARGET_WIDTH,
        'layer': LAYER,
        'd_model': model.cfg.d_model
    }, output_file)

    console.print(f"[green]✓ Width {TARGET_WIDTH} complete![/green]")
    console.print(f"Total tokens processed: {total_tokens:,}")
    console.print(f"Char positions found: 0-{max_position}")
    console.print(f"Saved to: {output_file}")
    console.print(f"Top positions by token count: {sorted(position_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")

console.print(f"\n[bold green]{'='*60}[/bold green]")
console.print(f"[bold green]All widths processing complete![/bold green]")
console.print(f"[bold green]{'='*60}[/bold green]")

# %% Cell 4: Combine All Widths
"""
Load all width-specific means and aggregate by char_position (pooled across widths)
"""

console.print("\n[bold cyan]Combining all width results[/bold cyan]")

# Find all processed files (match configured LAYER)
width_files = sorted(results_dir.glob(f"width_*_layer{LAYER}_means.pt"))

if len(width_files) == 0:
    console.print("[red]✗ No width files found! Run Cell 3 first.[/red]")
else:
    console.print(f"Found {len(width_files)} width files")

    # Collect all (char_position, activation) pairs
    all_positions_data = defaultdict(lambda: {'sum': None, 'count': 0})

    for width_file in track(width_files, description="Loading width files"):
        data = torch.load(width_file, weights_only=False)
        width = data['width']
        means = data['means']  # [max_pos+1, d_model]
        counts = data['counts']  # [max_pos+1]

        # Add each position's data
        for pos in range(len(means)):
            if counts[pos] > 0:  # Only include positions with data
                if all_positions_data[pos]['sum'] is None:
                    all_positions_data[pos]['sum'] = means[pos] * counts[pos]
                else:
                    all_positions_data[pos]['sum'] += means[pos] * counts[pos]
                all_positions_data[pos]['count'] += counts[pos]

    # Compute combined means
    max_pos = max(all_positions_data.keys())
    combined_means = torch.zeros(max_pos + 1, means.shape[1])
    combined_counts = torch.zeros(max_pos + 1, dtype=torch.long)

    for pos, data in all_positions_data.items():
        if data['count'] > 0:
            combined_means[pos] = data['sum'] / data['count']
            combined_counts[pos] = data['count']

    # Save combined results
    combined_file = results_dir / f"combined_layer{LAYER}_means.pt"
    torch.save({
        'means': combined_means,
        'counts': combined_counts,
        'layer': LAYER,
        'num_widths': len(width_files),
        'd_model': means.shape[1]
    }, combined_file)

    console.print(f"[green]✓ Combined results saved[/green]")
    console.print(f"Total char positions: {(combined_counts > 0).sum()}")
    console.print(f"Max char position: {max_pos}")
    console.print(f"Total tokens across all widths: {combined_counts.sum():,}")
    console.print(f"Shape: {combined_means.shape}")
    console.print(f"Saved to: {combined_file}")

    # Show distribution of tokens per position
    nonzero_positions = combined_counts[combined_counts > 0]
    console.print(f"\nTokens per position stats:")
    console.print(f"  Mean: {nonzero_positions.float().mean():.0f}")
    console.print(f"  Median: {nonzero_positions.float().median():.0f}")
    console.print(f"  Min: {nonzero_positions.min()}")
    console.print(f"  Max: {nonzero_positions.max()}")

# %% Cell 5: Run PCA
"""
Fit PCA on combined means to reduce from d_model (2304) to 3 dimensions
"""

from sklearn.decomposition import PCA

console.print("\n[bold cyan]Running PCA[/bold cyan]")

combined_file = results_dir / f"combined_layer{LAYER}_means.pt"

if not combined_file.exists():
    console.print("[red]✗ Combined file not found! Run Cell 4 first.[/red]")
else:
    # Load combined means
    data = torch.load(combined_file, weights_only=False)
    combined_means = data['means']  # [max_pos+1, d_model]
    combined_counts = data['counts']  # [max_pos+1]

    # Filter to only positions with data AND exclude position 0
    valid_mask = (combined_counts > 0) & (torch.arange(len(combined_means)) > 0)
    valid_positions = torch.arange(len(combined_means))[valid_mask]
    valid_means = combined_means[valid_mask]  # [n_valid_positions, d_model]
    valid_counts = combined_counts[valid_mask]

    console.print(f"Input shape: {valid_means.shape}")
    console.print(f"Valid positions: {len(valid_positions)} (excluding position 0)")
    console.print(f"Position 0 excluded (likely BOS/special token)")

    # Fit PCA
    console.print("Fitting PCA...")
    pca = PCA(n_components=3, whiten=True)
    pca_coords = pca.fit_transform(valid_means.numpy())  # [n_valid_positions, 3]

    # Show explained variance
    console.print(f"[green]✓ PCA complete[/green]")
    console.print(f"\nExplained variance ratio:")
    console.print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    console.print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    console.print(f"  PC3: {pca.explained_variance_ratio_[2]:.4f} ({pca.explained_variance_ratio_[2]*100:.2f}%)")
    console.print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # Save PCA results
    pca_file = results_dir / f"pca_layer{LAYER}_3d.pt"
    torch.save({
        'pca_coords': pca_coords,
        'positions': valid_positions.numpy(),
        'counts': valid_counts.numpy(),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'layer': LAYER
    }, pca_file)

    console.print(f"\nSaved to: {pca_file}")
    console.print(f"PCA coords shape: {pca_coords.shape}")

# %% Cell 6: 3D Plotly Visualization
"""
Create interactive 3D scatter plot of PCA results, colored by char_position
"""

import plotly.graph_objects as go
import plotly.express as px
""
console.print("\n[bold cyan]Creating 3D visualization[/bold cyan]")

pca_file = results_dir / f"pca_layer{LAYER}_3d.pt"

if not pca_file.exists():
    console.print("[red]✗ PCA file not found! Run Cell 5 first.[/red]")
else:
    # Load PCA results
    data = torch.load(pca_file, weights_only=False)
    pca_coords = data['pca_coords']  # [n_positions, 3]
    positions = data['positions']  # [n_positions]
    counts = data['counts']  # [n_positions]
    explained_var = data['explained_variance_ratio']

    console.print(f"Loaded {len(positions)} data points")

    # Create 3D scatter plot with lines connecting points in order
    fig = go.Figure(data=[go.Scatter3d(
        x=pca_coords[:, 0],
        y=pca_coords[:, 1],
        z=pca_coords[:, 2],
        mode='lines+markers',
        line=dict(
            color=positions,  # Color lines by char_position
            colorscale='Viridis',
            width=2
        ),
        marker=dict(
            size=2,  # Small fixed size for all dots
            color=positions,  # Color by char_position
            colorscale='Viridis',
            colorbar=dict(title="Char Position"),
            opacity=0.8
        ),
        text=[f"Pos: {p}<br>Tokens: {c:,}" for p, c in zip(positions, counts)],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
    )])

    # Update layout
    fig.update_layout(
        title=f"Layer {LAYER} Residual Stream Activations - PCA (3D)<br>" +
              f"<sub>Explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, PC3={explained_var[2]:.2%} (total={explained_var.sum():.2%})</sub>",
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]:.2%})',
            yaxis_title=f'PC2 ({explained_var[1]:.2%})',
            zaxis_title=f'PC3 ({explained_var[2]:.2%})',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1000,
        height=800,
        hovermode='closest'
    )

    # Show plot
    fig.show()

    console.print(f"[green]✓ Visualization complete![/green]")
    console.print("Interactive plot opened in browser")

    # Save as HTML for later viewing
    html_file = results_dir / f"pca_layer{LAYER}_3d_plot.html"
    fig.write_html(str(html_file))
    console.print(f"Plot saved to: {html_file}")

# %%
