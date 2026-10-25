"""생성 모델 백엔드의 공통 인터페이스.

단일 모델 원칙: 실제 실행에서는 Brain-Diffuser 계열 하나 (VDVAE + CLIP ViT-L/14 + Versatile Diffusion)
만 쓴다. mock 백엔드는 같은 인터페이스를 가진 아주 작은 가짜 모델로, GPU 없이 파이프라인을 점검할 때만 쓴다.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


class Backend:
    name = "base"
    clip_dim: int = 768
    n_clip_tokens: int = 257     # CLS + 16x16 패치
    n_text_tokens: int = 77
    vdvae_dim: int = 91168

    # ---- 특징 추출 ------------------------------------------------------
    def vdvae_encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """(N, vdvae_dim) VDVAE 사후분포 잠재 (Brain-Diffuser 와 같은 31층)."""
        raise NotImplementedError

    def clip_vision(self, images: Sequence[Image.Image]) -> np.ndarray:
        """(N, n_clip_tokens, clip_dim). 0번 토큰이 단위 노름의 이미지 임베딩."""
        raise NotImplementedError

    def clip_text(self, captions: Sequence[Sequence[str]]) -> np.ndarray:
        """(N, n_text_tokens, clip_dim). 그림마다 여러 캡션의 평균."""
        raise NotImplementedError

    # ---- 생성 ------------------------------------------------------------
    def vdvae_decode(self, latents: np.ndarray, seed: int = 0) -> List[Image.Image]:
        """잠재 -> 64px 저해상 이미지 -> 512px 로 키운 PIL 목록."""
        raise NotImplementedError

    def generate(self, init_images: Sequence[Image.Image], clip_vision: np.ndarray,
                 clip_text: Optional[np.ndarray], seed: int = 0,
                 strength: Optional[float] = None, mixing: Optional[float] = None) -> List[Image.Image]:
        """Versatile Diffusion 2단계. init_images 에서 출발해 시각/텍스트 조건으로 확산.

        clip_text 가 None 이면 빈 프롬프트로 대체하고 시각 조건만 쓴다 (mixing=0).
        strength=1.0 이면 init 을 무시하고 순수 잡음에서 시작한다 (빈자리 후보 생성용).
        """
        raise NotImplementedError

    # ---- 편의 ------------------------------------------------------------
    def clip_embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """(N, clip_dim) 단위 노름 이미지 임베딩 (CLS 토큰)."""
        tok = self.clip_vision(images)
        cls = tok[:, 0]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True).clip(1e-8)

    @property
    def patch_side(self) -> int:
        return int(round(np.sqrt(self.n_clip_tokens - 1)))
