"""특징 추출과 인코더 입력 블록.

NSD 자극 그림에서 세 가지 특징을 뽑는다 (Brain-Diffuser 와 동일한 모델).
  vdvae       (N, 91168)        VDVAE 31층 잠재
  clipvision  (N, 257, 768)     CLIP ViT-L/14 이미지 토큰 (VD 조건)
  cliptext    (N, 77, 768)      COCO 캡션 5개의 CLIP 텍스트 토큰 평균 (VD 조건)

인코더(그림 -> 복셀)의 입력은 같은 모델의 특징에서 만든다.
  clip_cls         (N, 768)          CLS 토큰
  clip_patch_pool  (N, g*g*768)      16x16 패치를 g x g 로 평균 풀링
  vdvae            (N, 91168)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- 그림 읽기

def load_image(path: Path, size: int = 512, mode: str = "center_crop") -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if mode == "center_crop":
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    elif mode == "pad":
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (127, 127, 127))
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        im = canvas
    else:
        raise ValueError(mode)
    return im.resize((size, size), resample=Image.BICUBIC)


def batches(n: int, bs: int) -> Iterable[slice]:
    for a in range(0, n, bs):
        yield slice(a, min(a + bs, n))


# ---------------------------------------------------------------- 인코더 블록

def pool_grid(backend_patch_side: int, requested: int) -> int:
    g = min(int(requested), backend_patch_side)
    while backend_patch_side % g != 0:
        g -= 1
    return max(g, 1)


def clip_patch_pool(tokens: np.ndarray, grid: int) -> np.ndarray:
    n, t, d = tokens.shape
    side = int(round(np.sqrt(t - 1)))
    g = pool_grid(side, grid)
    p = tokens[:, 1:].reshape(n, side, side, d)
    k = side // g
    p = p.reshape(n, g, k, g, k, d).mean(axis=(2, 4))
    return p.reshape(n, g * g * d).astype(np.float32)


def encoder_blocks(clip_tokens: np.ndarray, vdvae: np.ndarray, names: Sequence[str], grid: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name == "clip_cls":
            out[name] = clip_tokens[:, 0].astype(np.float32)
        elif name == "clip_patch_pool":
            out[name] = clip_patch_pool(clip_tokens, grid)
        elif name == "vdvae":
            out[name] = vdvae.astype(np.float32)
        else:
            raise ValueError(f"알 수 없는 인코더 블록: {name}")
    return out


# ---------------------------------------------------------------- 경로

def feature_dir(cfg) -> Path:
    d = cfg.get_path("paths.features") / f"subj{int(cfg.nsd.subject):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(cfg) -> Path:
    return cfg.get_path("paths.processed") / f"subj{int(cfg.nsd.subject):02d}"


def load_feature(cfg, split: str, name: str, mmap: bool = True) -> np.ndarray:
    return np.load(feature_dir(cfg) / f"{split}_{name}.npy", mmap_mode="r" if mmap else None)


# ---------------------------------------------------------------- NSD 특징 추출

def extract_nsd_features(cfg, backend, which: Sequence[str], log=print) -> None:
    out = feature_dir(cfg)
    pdir = processed_dir(cfg)
    bs = int(cfg.features.batch_size)
    for split in ("train", "test"):
        stim = np.load(pdir / f"{split}_stim.npy", mmap_mode="r")
        n = stim.shape[0]
        if "vdvae" in which:
            arr = np.lib.format.open_memmap(out / f"{split}_vdvae.npy", mode="w+", dtype=np.float32,
                                            shape=(n, backend.vdvae_dim))
            for sl in batches(n, bs):
                arr[sl] = backend.vdvae_encode([Image.fromarray(np.asarray(x)) for x in stim[sl]])
                log(f"[features] {split} vdvae {sl.stop}/{n}")
            arr.flush()
            del arr
        if "clipvision" in which:
            arr = np.lib.format.open_memmap(out / f"{split}_clipvision.npy", mode="w+", dtype=np.float32,
                                            shape=(n, backend.n_clip_tokens, backend.clip_dim))
            for sl in batches(n, bs):
                arr[sl] = backend.clip_vision([Image.fromarray(np.asarray(x)) for x in stim[sl]])
                log(f"[features] {split} clipvision {sl.stop}/{n}")
            arr.flush()
            del arr
        if "cliptext" in which:
            caps = np.load(pdir / f"{split}_cap.npy", allow_pickle=True)
            arr = np.lib.format.open_memmap(out / f"{split}_cliptext.npy", mode="w+", dtype=np.float32,
                                            shape=(n, backend.n_text_tokens, backend.clip_dim))
            for sl in batches(n, bs):
                arr[sl] = backend.clip_text([list(c) for c in caps[sl]])
                log(f"[features] {split} cliptext {sl.stop}/{n}")
            arr.flush()
            del arr
    if "clipvision" in which:
        build_encoder_clip_blocks(cfg, log)


def build_encoder_clip_blocks(cfg, log=print) -> None:
    """clipvision 토큰에서 인코더용 clip_cls, clip_patch_pool 블록을 미리 만들어 둔다."""
    out = feature_dir(cfg)
    grid = int(cfg.brain.encoder.clip_patch_pool_grid)
    bs = 512
    for split in ("train", "test"):
        tok = np.load(out / f"{split}_clipvision.npy", mmap_mode="r")
        n = tok.shape[0]
        cls = np.lib.format.open_memmap(out / f"{split}_enc_clip_cls.npy", mode="w+", dtype=np.float32,
                                        shape=(n, tok.shape[2]))
        first = clip_patch_pool(np.asarray(tok[:1]), grid)
        pool = np.lib.format.open_memmap(out / f"{split}_enc_clip_patch_pool.npy", mode="w+", dtype=np.float32,
                                         shape=(n, first.shape[1]))
        for sl in batches(n, bs):
            t = np.asarray(tok[sl], dtype=np.float32)
            cls[sl] = t[:, 0]
            pool[sl] = clip_patch_pool(t, grid)
        cls.flush()
        pool.flush()
        log(f"[features] {split} encoder clip blocks: cls {cls.shape}, patch_pool {pool.shape}")


def feature_stats(cfg, names: Sequence[str] = ("vdvae", "clipvision", "cliptext"), log=print) -> None:
    """학습셋 특징의 차원별 평균과 표준편차 (Brain-Diffuser 의 재정규화 목표값, ddof=0)."""
    out = feature_dir(cfg)
    stats = {}
    for name in names:
        X = np.load(out / f"train_{name}.npy", mmap_mode="r")
        flat = X.reshape(X.shape[0], -1)
        mean = np.zeros(flat.shape[1], dtype=np.float64)
        sq = np.zeros(flat.shape[1], dtype=np.float64)
        for sl in batches(flat.shape[0], 256):
            b = np.asarray(flat[sl], dtype=np.float64)
            mean += b.sum(0)
            sq += (b ** 2).sum(0)
        mean /= flat.shape[0]
        std = np.sqrt(np.maximum(sq / flat.shape[0] - mean ** 2, 0.0))
        std[std == 0] = 1.0
        stats[f"{name}_mean"] = mean.astype(np.float32)
        stats[f"{name}_std"] = std.astype(np.float32)
        log(f"[features] stats {name}: dim {flat.shape[1]}")
    np.savez(out / "train_feature_stats.npz", **stats)
