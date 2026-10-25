"""GPU 없이 파이프라인을 점검하기 위한 아주 작은 가짜 백엔드.

차원만 작고 인터페이스는 Brain-Diffuser 백엔드와 같다. 결과 이미지는 의미가 없다.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .base import Backend


def _img_array(im: Image.Image, side: int) -> np.ndarray:
    return np.asarray(im.convert("RGB").resize((side, side), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def _to_img(arr: np.ndarray, size: int = 512) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((arr * 255).astype(np.uint8))
    return im.resize((size, size), resample=Image.BICUBIC)


class MockBackend(Backend):
    name = "mock"
    clip_dim = 8
    n_clip_tokens = 5      # CLS + 2x2 패치
    n_text_tokens = 3
    vdvae_dim = 48         # 4x4x3

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed + 12345)
        self.P_cls = rng.standard_normal((self.clip_dim, 8 * 8 * 3)).astype(np.float32) / np.sqrt(192)
        self.P_patch = rng.standard_normal((self.clip_dim, 4 * 4 * 3)).astype(np.float32) / np.sqrt(48)
        self.P_cls_pinv = np.linalg.pinv(self.P_cls).astype(np.float32)
        self._rng = np.random.default_rng(seed)

    def vdvae_encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = np.stack([_img_array(im, 4).reshape(-1) for im in images])
        return (out + 0.01 * self._rng.standard_normal(out.shape)).astype(np.float32)

    def vdvae_decode(self, latents: np.ndarray, seed: int = 0) -> List[Image.Image]:
        latents = np.atleast_2d(latents)
        return [_to_img(z.reshape(4, 4, 3)) for z in latents]

    def clip_vision(self, images: Sequence[Image.Image]) -> np.ndarray:
        toks = []
        for im in images:
            a = _img_array(im, 8)
            cls = self.P_cls @ a.reshape(-1)
            patches = [self.P_patch @ a[r:r + 4, c:c + 4].reshape(-1) for r in (0, 4) for c in (0, 4)]
            t = np.stack([cls] + patches)
            toks.append(t / max(np.linalg.norm(cls), 1e-8))
        return np.stack(toks).astype(np.float32)

    def clip_text(self, captions: Sequence[Sequence[str]]) -> np.ndarray:
        out = []
        for caps in captions:
            vecs = []
            for c in caps:
                h = int(hashlib.md5(str(c).encode()).hexdigest()[:8], 16)
                vecs.append(np.random.default_rng(h).standard_normal((self.n_text_tokens, self.clip_dim)))
            out.append(np.mean(vecs, axis=0) if vecs else np.zeros((self.n_text_tokens, self.clip_dim)))
        return np.stack(out).astype(np.float32)

    def generate(self, init_images: Sequence[Image.Image], clip_vision: np.ndarray,
                 clip_text: Optional[np.ndarray], seed: int = 0,
                 strength: Optional[float] = None, mixing: Optional[float] = None) -> List[Image.Image]:
        strength = 0.75 if strength is None else float(strength)
        outs = []
        for i, init in enumerate(init_images):
            rng = np.random.default_rng(seed + i)
            target = (self.P_cls_pinv @ clip_vision[i, 0]).reshape(8, 8, 3)
            target = (target - target.min()) / max(float(np.ptp(target)), 1e-6)
            base = _img_array(init, 8) if strength < 1.0 else rng.random((8, 8, 3)).astype(np.float32)
            mix = (1 - strength) * base + strength * (0.5 * target + 0.5 * base)
            mix = mix + 0.03 * rng.standard_normal(mix.shape)
            outs.append(_to_img(mix))
        return outs
