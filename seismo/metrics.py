"""사슬의 각 단계에서 기록하는 지표."""
from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _gray(im: Image.Image, side: int) -> np.ndarray:
    return np.asarray(im.convert("L").resize((side, side), resample=Image.BILINEAR), dtype=np.float64) / 255.0


def pixel_corr(a: Image.Image, b: Image.Image, side: int = 64) -> float:
    return pearson(_gray(a, side), _gray(b, side))


def ssim(a: Image.Image, b: Image.Image, side: int = 256) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(_gray(a, side), _gray(b, side), data_range=1.0))


def step_metrics(img: Image.Image, origin: Image.Image, prev: Image.Image,
                 cls: np.ndarray, cls_origin: np.ndarray, cls_prev: np.ndarray) -> Dict[str, float]:
    return dict(
        clip_sim_to_origin=cosine(cls, cls_origin),
        clip_sim_to_prev=cosine(cls, cls_prev),
        ssim_to_origin=ssim(img, origin),
        pixcorr_to_origin=pixel_corr(img, origin),
    )
