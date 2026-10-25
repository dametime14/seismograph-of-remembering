from __future__ import annotations

from .base import Backend


def get_backend(cfg, need=("vdvae", "vd")) -> Backend:
    """설정의 backend 이름으로 백엔드를 만든다. need 로 필요한 부분만 불러 메모리를 아낀다."""
    name = str(cfg.backend).lower()
    if name == "mock":
        from .mock import MockBackend
        return MockBackend(seed=int(cfg.project.seed))
    if name == "braindiffuser":
        from .braindiffuser import BrainDiffuserBackend
        return BrainDiffuserBackend(cfg, need=need)
    raise ValueError(f"알 수 없는 backend: {name}")


__all__ = ["Backend", "get_backend"]
