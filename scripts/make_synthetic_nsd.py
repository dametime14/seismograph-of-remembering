"""mock 백엔드 점검용 가짜 자료를 만든다 (NSD 형식의 작은 배열, 그림 2장, 감정 표).

  python scripts/make_synthetic_nsd.py --out outputs/mock_data
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.backends.mock import MockBackend


def smooth_image(rng, side=64):
    a = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(a).resize((side, side), resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/mock_data")
    p.add_argument("--n-train", type=int, default=160)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--n-voxels", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    proc = out / "processed" / "subj01"
    proc.mkdir(parents=True, exist_ok=True)
    be = MockBackend(seed=0)
    words = ["sea", "dancer", "drapery", "wind", "column", "child", "fruit", "basket", "light", "stone"]

    W_true = rng.standard_normal((be.vdvae_dim + be.clip_dim * be.n_clip_tokens, args.n_voxels)) * 0.3
    for split, n in (("train", args.n_train), ("test", args.n_test)):
        stim = np.stack([smooth_image(rng) for _ in range(n)])
        ims = [Image.fromarray(s) for s in stim]
        f = np.concatenate([be.vdvae_encode(ims), be.clip_vision(ims).reshape(n, -1)], axis=1)
        fmri = (f - f.mean(0)) @ W_true + 0.5 * rng.standard_normal((n, args.n_voxels))
        np.save(proc / f"{split}_stim.npy", stim)
        np.save(proc / f"{split}_fmri.npy", (fmri * 300).astype(np.float32))
        caps = np.array([[" ".join(rng.choice(words, 4)) for _ in range(5)] for _ in range(n)])
        np.save(proc / f"{split}_cap.npy", caps)
        log(f"[synthetic] {split}: {n} images, fmri {fmri.shape}")

    arts = out / "artworks"
    arts.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        name = f"mock_art_{i}"
        Image.fromarray(smooth_image(rng, 256)).save(arts / f"{name}.png")
        rows.append(dict(id=name, file=f"{name}.png", title=f"Mock {i}", artist="none", year="", source_url="",
                         license="synthetic"))
    with open(arts / "artworks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    emo = out / "emotion"
    emo.mkdir(parents=True, exist_ok=True)
    n_vid = 120
    ids = np.arange(1, n_vid + 1)
    emb = rng.standard_normal((n_vid, be.clip_dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    logits = emb @ rng.standard_normal((be.clip_dim, 34)) * 2
    scores = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    np.savez(emo / "clip_video_embeddings.npz", ids=ids, emb=emb, ok=np.ones(n_vid, dtype=bool))
    with open(emo / "cowen_horikawa_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stimulus_num", "stim_num_int"] + [f"score_{i}" for i in range(34)])
        for sid, sc in zip(ids, scores):
            w.writerow([f"stimulus_{sid}", sid] + [f"{v:.5f}" for v in sc])

    (out / "panel.yaml").write_text(
        "name: mock panel\nslug: mock\ncondition: brain\nsigma: 0.5\nimages: [mock_art_0, mock_art_1, mock_art_2]\n")
    log(f"[synthetic] done -> {out}")


if __name__ == "__main__":
    main()
