"""
Visualize linebreak metadata for a given width and sequence.

Produces an interactive Plotly HTML with:
- char_count (on-line position after each token)
- vertical markers at newline tokens
- alternating background shading per line (derived directly from contains_newline)

Also prints a compact table of the first N tokens with token_id, decoded text (optional),
char_count, and newline flag.

Usage:
  python visualize_metadata.py --width 40 --seq 0 \
      --data-dir linebreak_data --out-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from rich.console import Console
from rich.table import Table
from typing import Tuple, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import html
console = Console()


def load_data(data_dir: Path, width: int, seq_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Load rows for a single sequence from saved files only.

    Required files (all within data_dir):
      - linebreak_width_{width}_tokens.pt
      - linebreak_width_{width}_char_count.pt
      - linebreak_width_{width}_contains_newline.pt
    Optional:
      - linebreak_width_{width}.txt (for raw text preview)
    """
    tokens_file = data_dir / f"linebreak_width_{width}_tokens.pt"
    char_file = data_dir / f"linebreak_width_{width}_char_count.pt"
    contains_file = data_dir / f"linebreak_width_{width}_contains_newline.pt"
    text_file = data_dir / f"linebreak_width_{width}.txt"

    for f in [tokens_file, char_file, contains_file]:
        if not f.exists():
            console.print(f"[red]Missing required file:[/red] {f}")
            sys.exit(1)

    tokens = torch.load(tokens_file, weights_only=False)
    char_counts = torch.load(char_file, weights_only=False)
    contains_nl = torch.load(contains_file, weights_only=False)

    if seq_idx < 0 or seq_idx >= tokens.shape[0]:
        console.print(f"[red]seq index out of range[/red]: 0..{tokens.shape[0]-1}")
        sys.exit(1)

    # Determine true (unpadded) sequence length
    true_len = None
    json_file = data_dir / f"linebreak_width_{width}_tokens.json"
    if json_file.exists():
        try:
            import json as _json
            seqs = _json.loads(json_file.read_text())
            if 0 <= seq_idx < len(seqs):
                true_len = len(seqs[seq_idx])
        except Exception:
            true_len = None
    if true_len is None:
        # Fallback: strip trailing zeros in token ids (padding convention)
        row = tokens[seq_idx].tolist()
        i = len(row)
        while i > 0 and int(row[i - 1]) == 0:
            i -= 1
        true_len = max(0, i)

    # Slice to true length
    tokens_row = tokens[seq_idx][:true_len]
    char_row = char_counts[seq_idx][:true_len]
    contains_row = contains_nl[seq_idx][:true_len]

    # Extract raw text sequence (if available)
    raw_text = ""
    if text_file.exists():
        try:
            content = text_file.read_text(encoding="utf-8")
            parts = content.split("\n\n=====\n\n")
            if 0 <= seq_idx < len(parts):
                raw_text = parts[seq_idx]
        except Exception:
            pass

    return tokens_row, char_row, contains_row, raw_text


def compute_line_numbers(is_newline: List[int]) -> List[int]:
    """Compute line_number per token as cumulative count of newline tokens seen so far."""
    ln = 0
    out = []
    for nl in is_newline:
        out.append(ln)
        if int(nl) == 1:
            ln += 1
    return out


def try_decode_tokens(token_ids: torch.Tensor) -> List[str] | None:
    """Attempt to decode tokens with Gemma; uses only token IDs from saved data."""
    try:
        from linebreak_utils import get_gemma_tokenizer
        tok = get_gemma_tokenizer()
        ids = token_ids.tolist()
        decoded = []
        for tid in ids:
            if tid == 0:
                decoded.append("⟂")
            else:
                try:
                    decoded.append(tok.decode([int(tid)]))
                except Exception:
                    decoded.append("")
        return decoded
    except Exception as e:
        console.print(f"[yellow]Tokenizer unavailable; proceeding without decoded tokens ({e})[/yellow]")
        return None


def build_plot(char_count_after: List[int], is_newline: List[int], line_width: int):
    n = len(char_count_after)
    x = list(range(n))

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Compute simple line numbers for background shading
    line_no = compute_line_numbers(is_newline)

    # Alternating background per line segment
    # Find contiguous spans of constant line_no
    spans = []
    if n > 0:
        start = 0
        for i in range(1, n):
            if line_no[i] != line_no[i - 1]:
                spans.append((start, i - 1, line_no[i - 1]))
                start = i
        spans.append((start, n - 1, line_no[-1]))

    for i, (s, e, ln) in enumerate(spans):
        fig.add_vrect(
            x0=s - 0.5, x1=e + 0.5,
            fillcolor="#f0f0f0" if (ln % 2 == 0) else "#ffffff",
            opacity=0.35, layer="below", line_width=0,
        )

    # Plot char_count (after each token)
    fig.add_trace(go.Scatter(x=x, y=char_count_after, mode="lines+markers", name="char_count",
                             line=dict(color="#1f77b4"), marker=dict(size=4)))

    # Newline markers
    nl_x = [i for i, nl in enumerate(is_newline) if nl == 1]
    if nl_x:
        fig.add_trace(go.Scatter(x=nl_x, y=[0]*len(nl_x), mode="markers", name="newline token",
                                 marker=dict(symbol="line-ns", size=10, color="#2ca02c")))

    # Line width reference
    fig.add_hline(y=line_width, line_dash="dot", line_color="#999", annotation_text=f"k={line_width}",
                  annotation_position="top right")

    fig.update_layout(
        title=f"Current char position vs token index (k={line_width})",
        xaxis_title="token index",
        yaxis_title="char count on line",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # y-axis bounds
    ymax = max(max(char_count_after or [0]), line_width)
    fig.update_yaxes(range=[-1, max(5, ymax + 2)])

    return fig


def build_token_view_html(token_texts: List[str], line_width: int, *, char_after: List[int], label_stride: int = 1) -> str:
    """Render tokens with colored backgrounds and numeric labels.

    Uses only saved features (char_after) for labels. Newlines in token_texts
    render naturally via CSS whitespace preservation. Newline tokens are treated
    like any other token — no special styling.
    """
    palette = [
        "#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f",
        "#90be6d", "#43aa8b", "#577590", "#277da1", "#9b5de5",
        "#f15bb5", "#fee440", "#00bbf9", "#00f5d4", "#bc6c25",
    ]
    css = f"""
    <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .wrap {{ position: relative; white-space: pre; line-height: 1.8; }}
    .kbar {{ position: absolute; left: calc({line_width}ch); top: 0; bottom: 0; width: 0; border-left: 1px dashed #888; opacity: 0.6; pointer-events: none; }}
    .tok {{ position: relative; display: inline; padding: 0; margin: 0; }}
    .tok .label {{ position: absolute; top: -1.1em; left: 0; font-size: 0.75em; color: #222; background: rgba(255,255,255,0.6); padding: 0 2px; border-radius: 2px; }}
    </style>
    """
    parts = []
    for i, text in enumerate(token_texts):
        col = palette[i % len(palette)]
        bg = col + "22"  # light alpha overlay
        label_html = ""
        if label_stride > 0 and (i % label_stride == 0):
            label_html = f"<span class='label' style='color:{col}'>{int(char_after[i])}</span>"
        token_html = (
            f"<span class='tok' style='background-color:{bg}; border-bottom:2px solid {col};'>"
            f"{label_html}{html.escape(text)}"
            f"</span>"
        )
        parts.append(token_html)
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        + css + "</head><body>"
        + f"<div class='wrap'><div class='kbar'></div>{''.join(parts)}</div>"
        + "</body></html>"
    )
    return html_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="linebreak_data")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--preview", type=int, default=40, help="How many tokens to print in table preview")
    parser.add_argument("--decode", action="store_true", help="Attempt to decode tokens with Gemma tokenizer for preview table")
    parser.add_argument("--label-stride", type=int, default=1, help="Show label every N tokens in text preview (1 = every token)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_row, char_row, is_nl_row, raw_text = load_data(data_dir, args.width, args.seq)
    # Convert to Python lists for plotting
    char_after = [int(x) for x in char_row.tolist()]
    is_nl = [int(x) for x in is_nl_row.tolist()]
    line_width = args.width

    decoded = try_decode_tokens(tokens_row) if args.decode else None

    # Print a compact table
    table = Table(title=f"Feature preview — width={args.width} seq={args.seq}")
    table.add_column("idx", justify="right")
    table.add_column("token_id", justify="right")
    table.add_column("decoded", overflow="fold")
    table.add_column("char_count")
    table.add_column("is_nl")

    token_ids = tokens_row.tolist()
    preview_n = min(args.preview, len(token_ids))
    for i in range(preview_n):
        d = decoded[i] if decoded is not None else ""
        table.add_row(
            str(i),
            str(token_ids[i]),
            d,
            str(char_after[i]),
            "✓" if is_nl[i] else "",
        )
    console.print(table)

    if raw_text:
        console.print("\n[b]Wrapped text (first ~300 chars):[/b]")
        console.print(raw_text[:300] + ("…" if len(raw_text) > 300 else ""))

    # Plotly curve view
    fig = build_plot(char_after, is_nl, line_width)
    out_file = out_dir / f"metadata_viz_width_{args.width}_seq_{args.seq}.html"
    fig.write_html(str(out_file), include_plotlyjs="inline")
    console.print(f"[green]✓[/green] Saved plot: {out_file}")

    # Token-colored text view with labels (uses only saved features + optional decoding)
    token_texts = try_decode_tokens(tokens_row)
    if token_texts is None:
        # Fallback: show token IDs; inject a newline character when contains_newline=1
        tids = tokens_row.tolist()
        token_texts = []
        for i, tid in enumerate(tids):
            s = "⟂" if int(tid) == 0 else str(int(tid))
            if is_nl[i]:
                s += "\n"
            token_texts.append(s)
    textviz_html = build_token_view_html(
        token_texts=token_texts,
        line_width=line_width,
        char_after=char_after,
        label_stride=args.label_stride,
    )
    text_file = out_dir / f"metadata_textviz_width_{args.width}_seq_{args.seq}.html"
    text_file.write_text(textviz_html, encoding="utf-8")
    console.print(f"[green]✓[/green] Saved text viz: {text_file}")


if __name__ == "__main__":
    main()
