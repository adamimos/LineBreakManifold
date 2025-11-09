# %%
"""
VSCode cell to instantiate Gemma-2-27b with TransformerLens.
Requires a high-memory CUDA GPU and HuggingFace access for google/gemma-2-27b.
"""

from transformer_lens import HookedTransformer
import torch

# Choose a sensible default device/dtype
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

# Gemma-2-27b on Hugging Face is "google/gemma-2-27b"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32
model = HookedTransformer.from_pretrained(
    "google/gemma-2-27b",
    device=DEVICE,
    dtype=DTYPE,
)

print(f"Loaded Gemma-2-27b as HookedTransformer on {DEVICE} (dtype={DTYPE}).")
# %%
"""
Streaming DataLoader: yields only fixed-length sequences that
- start at the beginning of each sample, and
- exactly fill the model context window (others are skipped).
Iterates over the entire dataset across provided widths.
"""

from typing import Iterator, Sequence, Dict
from pathlib import Path


class FixedContextStream(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str | Path, widths: int | Sequence[int], context_length: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        # Accept single int or sequence of ints
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

        # Determine true lengths from JSON if available; else trim trailing zeros
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
                row = t[i].tolist()
                j = len(row)
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


# Optional: randomly interleave widths per sample
import random

class RandomWidthStream(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str | Path, widths: int | Sequence[int], context_length: int, seed: int | None = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.widths = [int(widths)] if isinstance(widths, int) else [int(x) for x in widths]
        self.L = int(context_length)
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        active = []  # list of dicts per width: {w, t, c, n, idxs, pos}
        for w in self.widths:
            t_file = self.data_dir / f"linebreak_width_{w}_tokens.pt"
            c_file = self.data_dir / f"linebreak_width_{w}_char_count.pt"
            n_file = self.data_dir / f"linebreak_width_{w}_contains_newline.pt"
            if not (t_file.exists() and c_file.exists() and n_file.exists()):
                continue
            t = torch.load(t_file, weights_only=False)
            c = torch.load(c_file, weights_only=False)
            n = torch.load(n_file, weights_only=False)
            # lengths
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


# Example usage: Iterate through a few batches, printing out metadata for each
DATA_DIR = "linebreak_data"
STREAM_WIDTHS = list(range(15, 151, 5))
CTX = model.cfg.n_ctx
stream_ds = FixedContextStream(DATA_DIR, STREAM_WIDTHS, CTX)
stream_loader = torch.utils.data.DataLoader(stream_ds, batch_size=64)

# Or randomly interleave widths:
# rand_stream = RandomWidthStream(DATA_DIR, STREAM_WIDTHS, CTX, seed=42)
# stream_loader = torch.utils.data.DataLoader(rand_stream, batch_size=32)


# %%
# Iterate through a fixed number of batches and run model in run_with_cache, print cache keys per batch,
# using the Rich console progress bar.
# This version processes exactly MAX_BATCHES batches if available, otherwise as many as exist in the stream.

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.console import Console
console = Console()
# We'll process up to MAX_BATCHES batches, or all available if fewer
MAX_BATCHES = 10

from itertools import islice

# %%
"""
Running average of activation vectors per char_count value (online update).
Uses Welford-style updates so the mean is correct at every step without
needing the final count.
"""

from collections import defaultdict

# Choose which layer to aggregate
LAYER = 1  # layer index to analyze
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
# Restrict aggregation to char_count strictly between 30 and 120
MIN_CHAR = 31
MAX_CHAR = 119



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
            delta = x - self.mean
            self.mean.add_(delta / self.n)

stats: dict[int, OnlineMean] = {}

# Use random interleaved widths for averaging
rand_stream = RandomWidthStream(DATA_DIR, STREAM_WIDTHS, CTX, seed=42)
stream_loader = torch.utils.data.DataLoader(rand_stream, batch_size=32)

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

# We'll use a progress bar over all batches, up to MAX_BATCHES if available
with Progress(
    SpinnerColumn(),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
    transient=True,
    console=console
) as progress:
    task = progress.add_task("[cyan]Processing batches", total=MAX_BATCHES)
    for i, batch in enumerate(islice(stream_loader, MAX_BATCHES)):
        print(f"Processing batch {i+1} of {MAX_BATCHES}")
        tokens = batch["tokens"].to(model.cfg.device)
        char_count = batch["char_count"]  # CPU
        contains_nl = batch["contains_newline"]  # CPU
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=HOOK_NAME,
                stop_at_layer=LAYER + 1,
                return_type="logits",
            )
        acts = cache[HOOK_NAME]  # [B, T, d_model]
        d_model = int(acts.shape[-1])

        for b in range(acts.shape[0]):
            for t in range(acts.shape[1]):
                # Skip any token whose span contains a newline
                if int(contains_nl[b, t]) != 0:
                    continue
                k = int(char_count[b, t])
                if k < MIN_CHAR or k > MAX_CHAR:
                    continue
                if k not in stats:
                    stats[k] = OnlineMean(d_model)
                v = acts[b, t]
                stats[k].update(v)
        progress.update(task, advance=1)
        if (i+1) >= MAX_BATCHES:
            break

charcount_means = {k: s.mean.clone() for k, s in stats.items() if s.n > 0}
print({
    "buckets": len(charcount_means),
    "example_keys": sorted(list(charcount_means.keys()))[:10],
    "vec_dim": next(iter(charcount_means.values())).shape[0] if charcount_means else 0,
})
# %%
"""
PCA (3D) of per-char_count mean activation vectors and interactive visualization.
"""

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

if 'charcount_means' not in globals() or not charcount_means:
    raise RuntimeError("charcount_means not found. Run the aggregation cell first.")

# Respect the same char_count window as aggregation
_minc, _maxc = MIN_CHAR, MAX_CHAR
keys = sorted(k for k in charcount_means.keys() if (_minc <= k <= _maxc))
X = np.stack([charcount_means[k].numpy() for k in keys], axis=0)  # [K, d]

# Compute PCA with 3 components (PC1–PC3 only)
pca = PCA(n_components=3, svd_solver='auto', random_state=0)
Z = pca.fit_transform(X)  # [K, 3]
evr = pca.explained_variance_ratio_

fig = go.Figure()

# Anthropic brand palette
dark = "#141413"; light = "#faf9f5"; mid = "#b0aea5"; light_gray = "#e8e6dc"
orange = "#d97757"; blue = "#6a9bcc"; green = "#788c5d"
brand_scale = [[0.0, orange], [0.5, blue], [1.0, green]]

_cmin = _minc if _minc is not None else (keys[0] if keys else 0)
_cmax = _maxc if _maxc is not None else (keys[-1] if keys else 1)

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
            colorbar=dict(
                title=dict(text="char_count", font=dict(family="Poppins, Arial, sans-serif", color=dark)),
                bgcolor=light,
                tickcolor=dark,
                tickfont=dict(family="Lora, Georgia, serif", color=dark),
            ),
            line=dict(color=light, width=0.6),
        ),
        text=[f"k={k}" for k in keys],
        hovertemplate=(
            "<b>%{text}</b><br>"
            + "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>"
        ),
        name="char_count means",
    )
)

fig.update_layout(
    title=dict(
        text=f"PCA of Char Count Means — EVR: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f}",
        font=dict(family="Poppins, Arial, sans-serif", color=dark, size=20),
        x=0.02,
    ),
    font=dict(family="Lora, Georgia, serif", color=dark, size=13),
    paper_bgcolor=light,
    scene=dict(
        xaxis=dict(
            title=dict(text="PC1", font=dict(family="Poppins, Arial, sans-serif", color=dark)),
            gridcolor=light_gray, zerolinecolor=mid, showbackground=True, backgroundcolor=light,
            tickfont=dict(family="Lora, Georgia, serif", color=dark),
        ),
        yaxis=dict(
            title=dict(text="PC2", font=dict(family="Poppins, Arial, sans-serif", color=dark)),
            gridcolor=light_gray, zerolinecolor=mid, showbackground=True, backgroundcolor=light,
            tickfont=dict(family="Lora, Georgia, serif", color=dark),
        ),
        zaxis=dict(
            title=dict(text="PC3", font=dict(family="Poppins, Arial, sans-serif", color=dark)),
            gridcolor=light_gray, zerolinecolor=mid, showbackground=True, backgroundcolor=light,
            tickfont=dict(family="Lora, Georgia, serif", color=dark),
        ),
    ),
    hoverlabel=dict(bgcolor=light_gray, font=dict(family="Lora, Georgia, serif", color=dark)),
    margin=dict(l=10, r=10, t=50, b=10),
    height=640,
)

# Helper to interpolate brand colors across [orange -> blue -> green]
def _hex_to_rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0,2,4))
def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def _lerp(a,b,t):
    return tuple(int(round(a[i] + (b[i]-a[i])*t)) for i in range(3))
_o,_b,_g = _hex_to_rgb(orange), _hex_to_rgb(blue), _hex_to_rgb(green)
def brand_color_for_k(k, kmin, kmax):
    if kmax==kmin:
        t=0.0
    else:
        t = (k - kmin)/(kmax - kmin)
    if t <= 0.5:
        c = _lerp(_o, _b, t/0.5)
    else:
        c = _lerp(_b, _g, (t-0.5)/0.5)
    return _rgb_to_hex(c)

# Add per-segment colored connecting lines
if len(keys) >= 2:
    kmin, kmax = (_cmin, _cmax)
    for i in range(len(keys)-1):
        k_mid = 0.5*(keys[i] + keys[i+1])
        c = brand_color_for_k(k_mid, kmin, kmax)
        fig.add_trace(
            go.Scatter3d(
                x=[Z[i,0], Z[i+1,0]], y=[Z[i,1], Z[i+1,1]], z=[Z[i,2], Z[i+1,2]],
                mode='lines',
                line=dict(color=c, width=3),
                hoverinfo='skip',
                showlegend=False,
            )
        )

# Show and save
# Try to show inline; if nbformat is unavailable, fall back to opening in browser
try:
    fig.show()
except Exception:
    pio.renderers.default = "browser"
    try:
        fig.show()
    except Exception:
        pass
out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "charcount_pca_3d_pc1_3.html"
fig.write_html(str(out_file), include_plotlyjs="cdn")
print({"saved_pc1_3": str(out_file)})




# %%
