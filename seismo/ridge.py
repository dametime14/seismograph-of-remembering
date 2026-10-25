"""릿지 회귀 (커널 또는 쌍대 형태).

인공 뇌의 인코더와 디코더는 모두 선형 릿지다. Brain-Diffuser 는 sklearn.Ridge 를
토큰마다 따로 돌렸지만, 같은 입력과 같은 알파를 쓰는 다중 출력 릿지는 해가 같으므로
여기서는 그람 행렬 K = Xc Xc^T 의 고윳값 분해를 한 번만 하고 출력 열을 조각내어 푼다.

  W = Xc^T U diag(1 / (s + alpha)) U^T Yc     (쌍대 해, n_samples < n_features 일 때 유리)
  예측 = (x - x_mean) W + y_mean               (fit_intercept=True 인 sklearn.Ridge 와 동일)

큰 행렬은 numpy memmap 에서 열 조각 단위로 읽으므로 메모리에 다 올리지 않아도 된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import torch


def _dev(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[ridge] {device} 를 쓸 수 없어 cpu 로 계산한다")
        return torch.device("cpu")
    return torch.device(device)


def column_chunks(n_cols: int, chunk: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_cols, chunk):
        yield start, min(start + chunk, n_cols)


def column_mean(X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = np.empty(X.shape[1], dtype=np.float64)
    for a, b in column_chunks(X.shape[1], chunk):
        out[a:b] = np.asarray(X[:, a:b], dtype=np.float64).mean(0)
    return out


def column_std(X: np.ndarray, mean: np.ndarray, chunk: int = 8192, ddof: int = 1) -> np.ndarray:
    out = np.empty(X.shape[1], dtype=np.float64)
    for a, b in column_chunks(X.shape[1], chunk):
        block = np.asarray(X[:, a:b], dtype=np.float64) - mean[a:b]
        out[a:b] = np.sqrt((block ** 2).sum(0) / max(X.shape[0] - ddof, 1))
    out[out == 0] = 1.0
    return out


def gram(blocks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
         device: str = "cpu", chunk: int = 8192) -> torch.Tensor:
    """K = sum_b w_b Z_b Z_b^T,  Z_b = (X_b - mean_b) / scale_b.

    blocks: (X_b, mean_b, scale_b, weight_b) 의 목록. X_b 는 memmap 이어도 된다.
    반환값은 float64 (n x n) 텐서.
    """
    dev = _dev(device)
    n = blocks[0][0].shape[0]
    K = torch.zeros((n, n), dtype=torch.float64, device=dev)
    for X, mean, scale, weight in blocks:
        for a, b in column_chunks(X.shape[1], chunk):
            Z = (np.asarray(X[:, a:b], dtype=np.float32) - mean[a:b].astype(np.float32)) / scale[a:b].astype(np.float32)
            Zt = torch.from_numpy(Z).to(dev)
            K += weight * (Zt @ Zt.T).double()
            del Zt
    return K


def eigh(K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    s, U = torch.linalg.eigh(K)
    s = torch.clamp(s, min=0.0)
    return s, U


def pearson_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    den[den == 0] = np.inf
    return (a * b).sum(0) / den


# --------------------------------------------------------------------------
# 디코더: fMRI (n x p) -> 특징 (n x q), 알파 고정 (Brain-Diffuser 값)
# --------------------------------------------------------------------------

@dataclass
class LinearMap:
    """y = (x - x_mean) @ W_T.T + y_mean.  W_T 는 (q x p) 로 저장한다 (행이 출력)."""
    W_T: np.ndarray        # (q, p)  memmap 가능
    x_mean: np.ndarray     # (p,)
    y_mean: np.ndarray     # (q,)

    @property
    def in_dim(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def out_dim(self) -> int:
        return int(self.y_mean.shape[0])

    def predict(self, X: np.ndarray, device: str = "cpu", chunk: int = 16384) -> np.ndarray:
        dev = _dev(device)
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        Xc = torch.from_numpy(X - self.x_mean.astype(np.float32)).to(dev)
        out = np.empty((X.shape[0], self.out_dim), dtype=np.float32)
        for a, b in column_chunks(self.out_dim, chunk):
            Wt = torch.from_numpy(np.asarray(self.W_T[a:b], dtype=np.float32)).to(dev)
            out[:, a:b] = (Xc @ Wt.T).cpu().numpy() + self.y_mean[a:b].astype(np.float32)
            del Wt
        return out

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if not (isinstance(self.W_T, np.memmap) and Path(self.W_T.filename) == prefix.with_suffix(".W_T.npy")):
            np.save(prefix.with_suffix(".W_T.npy"), np.asarray(self.W_T, dtype=np.float32))
        np.savez(prefix.with_suffix(".stats.npz"), x_mean=self.x_mean, y_mean=self.y_mean)

    @classmethod
    def load(cls, prefix: Path, mmap: bool = True) -> "LinearMap":
        prefix = Path(prefix)
        W_T = np.load(prefix.with_suffix(".W_T.npy"), mmap_mode="c" if mmap else None)
        st = np.load(prefix.with_suffix(".stats.npz"))
        return cls(W_T=W_T, x_mean=st["x_mean"], y_mean=st["y_mean"])


def fit_ridge_fixed_alpha(X: np.ndarray, Y: np.ndarray, alpha: float, out_prefix: Path,
                          device: str = "cpu", chunk: int = 8192,
                          log: Callable[[str], None] = print) -> LinearMap:
    """다중 출력 릿지 (알파 하나). 가중치는 out_prefix.W_T.npy 에 (q x p) memmap 으로 바로 쓴다."""
    dev = _dev(device)
    n, p = X.shape
    q = Y.shape[1]
    x_mean = X.mean(0).astype(np.float64)
    Xc = torch.from_numpy((X - x_mean).astype(np.float32)).to(dev)
    log(f"[ridge] gram {n}x{n}")
    K = (Xc @ Xc.T).double()
    s, U = eigh(K)
    del K
    filt = (1.0 / (s + alpha)).to(torch.float64)
    y_mean = column_mean(Y, chunk)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    W_T = np.lib.format.open_memmap(out_prefix.with_suffix(".W_T.npy"), mode="w+",
                                    dtype=np.float32, shape=(q, p))
    for a, b in column_chunks(q, chunk):
        Yc = torch.from_numpy(np.asarray(Y[:, a:b], dtype=np.float64) - y_mean[a:b]).to(dev)
        A = U @ (filt[:, None] * (U.T @ Yc))           # (n, c)
        W_T[a:b] = (A.T.float() @ Xc).cpu().numpy()    # (c, p)
        log(f"[ridge]   outputs {b}/{q}")
        del Yc, A
    W_T.flush()
    lm = LinearMap(W_T=W_T, x_mean=x_mean, y_mean=y_mean)
    np.savez(out_prefix.with_suffix(".stats.npz"), x_mean=x_mean, y_mean=y_mean, alpha=alpha)
    return lm


# --------------------------------------------------------------------------
# 인코더: 그림 특징 (여러 블록) -> 복셀, 복셀마다 알파를 검증셋으로 고른다
# --------------------------------------------------------------------------

def select_alpha_holdout(K: torch.Tensor, Y: np.ndarray, alphas: Sequence[float],
                         val_idx: np.ndarray, log: Callable[[str], None] = print
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """학습셋 안의 검증셋으로 출력(복셀)마다 가장 좋은 알파를 고른다.

    반환: best_alpha (T,), scores (len(alphas), T)  [검증 상관계수]
    """
    dev = K.device
    n = K.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    fit_idx = np.where(mask)[0]
    fi = torch.from_numpy(fit_idx).to(dev)
    vi = torch.from_numpy(np.asarray(val_idx)).to(dev)
    K_ff = K[fi][:, fi]
    K_vf = K[vi][:, fi]
    s, U = eigh(K_ff)
    Y_fit = Y[fit_idx].astype(np.float64)
    y_mean = Y_fit.mean(0)
    UtY = U.T @ torch.from_numpy(Y_fit - y_mean).to(dev)
    KvU = K_vf @ U
    Y_val = Y[val_idx].astype(np.float64)
    scores = np.zeros((len(alphas), Y.shape[1]), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        pred = (KvU @ ((1.0 / (s + alpha))[:, None] * UtY)).cpu().numpy() + y_mean
        scores[i] = pearson_columns(pred, Y_val)
        log(f"[ridge] alpha={alpha:g}  median r={np.nanmedian(scores[i]):.4f}")
    best = np.asarray(alphas)[np.nanargmax(np.nan_to_num(scores, nan=-1.0), axis=0)]
    return best, scores


def dual_coef_per_target_alpha(K: torch.Tensor, Y: np.ndarray, alpha_per_target: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """K (n x n, 중심화된 특징의 그람) 와 출력별 알파로 쌍대 계수 A (n x T) 를 푼다."""
    dev = K.device
    s, U = eigh(K)
    y_mean = Y.mean(0).astype(np.float64)
    UtY = U.T @ torch.from_numpy(Y.astype(np.float64) - y_mean).to(dev)
    A = np.zeros((K.shape[0], Y.shape[1]), dtype=np.float32)
    for alpha in np.unique(alpha_per_target):
        cols = np.where(alpha_per_target == alpha)[0]
        c = torch.from_numpy(cols).to(dev)
        A[:, cols] = (U @ ((1.0 / (s + float(alpha)))[:, None] * UtY[:, c])).float().cpu().numpy()
    return A, y_mean


def primal_from_dual(blocks: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
                     A: np.ndarray, out_path: Path, device: str = "cpu", chunk: int = 8192
                     ) -> np.ndarray:
    """W = Z^T A 를 블록별로 계산해 (P_total x T) memmap 으로 쓴다.

    Z 는 가중치가 곱해진 표준화 특징 [sqrt(w_b) Z_b]. 예측은 z(x) @ W + y_mean.
    """
    dev = _dev(device)
    P_total = sum(X.shape[1] for X, _, _, _ in blocks)
    W = np.lib.format.open_memmap(Path(out_path), mode="w+", dtype=np.float32, shape=(P_total, A.shape[1]))
    At = torch.from_numpy(A).to(dev)
    offset = 0
    for X, mean, scale, weight in blocks:
        for a, b in column_chunks(X.shape[1], chunk):
            Z = (np.asarray(X[:, a:b], dtype=np.float32) - mean[a:b].astype(np.float32)) / scale[a:b].astype(np.float32)
            Zt = torch.from_numpy(Z * np.float32(np.sqrt(weight))).to(dev)
            W[offset + a: offset + b] = (Zt.T @ At).cpu().numpy()
        offset += X.shape[1]
    W.flush()
    return W
