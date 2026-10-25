"""사슬을 한눈에 보는 밀착 인화(contact sheet)와 곡선."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .chain import read_log


def contact_sheet(artwork_dir: Path, steps: Sequence[int], out_path: Path, rep: int = 0,
                  thumb: int = 160, pad: int = 6) -> Optional[Path]:
    rows = []
    for cond_dir in sorted(p for p in artwork_dir.iterdir() if p.is_dir()):
        d = cond_dir / f"rep{rep:02d}"
        if not d.exists():
            continue
        tiles = []
        for t in steps:
            f = d / f"t{t:03d}.png"
            tiles.append(Image.open(f).convert("RGB").resize((thumb, thumb)) if f.exists() else None)
        rows.append((cond_dir.name, tiles))
    if not rows:
        return None
    label_w = 170
    W = label_w + len(steps) * (thumb + pad) + pad
    H = 28 + len(rows) * (thumb + pad) + pad
    sheet = Image.new("RGB", (W, H), (245, 243, 238))
    draw = ImageDraw.Draw(sheet)
    for j, t in enumerate(steps):
        draw.text((label_w + pad + j * (thumb + pad), 8), f"t = {t}", fill=(40, 40, 40))
    for i, (name, tiles) in enumerate(rows):
        y = 28 + pad + i * (thumb + pad)
        draw.text((8, y + thumb // 2 - 6), name, fill=(40, 40, 40))
        for j, tile in enumerate(tiles):
            if tile is not None:
                sheet.paste(tile, (label_w + pad + j * (thumb + pad), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def similarity_curves(artwork_dir: Path, out_path: Path, metric: str = "clip_sim_to_origin") -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond_dir in sorted(p for p in artwork_dir.iterdir() if p.is_dir()):
        curves = []
        for d in sorted(cond_dir.glob("rep*")):
            if (d / "log.jsonl").exists():
                curves.append([1.0] + [r[metric] for r in read_log(d)])
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = np.array([c[:L] for c in curves])
        m = arr.mean(0)
        ax.plot(range(L), m, label=cond_dir.name)
        if len(arr) > 1:
            ax.fill_between(range(L), arr.min(0), arr.max(0), alpha=0.15)
    ax.set_xlabel("t (passes through the brain)")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
