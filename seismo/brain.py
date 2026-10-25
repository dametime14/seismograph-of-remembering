"""인공 뇌: 인코더 E (그림 특징 -> 복셀) 와 디코더 D (복셀 -> 생성 모델 조건).

  E : [clip_cls, clip_patch_pool, vdvae]  ->  nsdgeneral 복셀 (z 점수)      커널 릿지, 복셀별 알파
  D : 복셀  ->  vdvae 잠재 / CLIP-V 토큰 / CLIP-T 토큰                     릿지, Brain-Diffuser 알파

두 단계가 모두 선형이므로 '뇌를 한 번 지나가는 것' 자체는 복셀 공간을 지나는 선형 병목이다.
사슬의 비선형성은 생성 모델(VDVAE 복원, VD 확산)과 생성된 그림을 다시 보는 단계에서 생긴다.

보정(calibration): Brain-Diffuser 는 테스트 예측 전체의 배치 평균/표준편차로 디코더 출력을
재정규화했다. 사슬은 그림을 한 장씩 처리하므로, NSD 테스트 982장을 E -> (+잡음) -> D 로 통과시켜
얻은 고정 통계로 대신한다. 잡음 크기(sigma)마다 따로 구한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import ridge
from .features import encoder_blocks, feature_dir, load_feature
from .nsd import load_fmri, zscore_fmri

DECODER_NAMES = ("vdvae", "clipvision", "cliptext")


def models_dir(cfg) -> Path:
    d = cfg.get_path("paths.models") / f"subj{int(cfg.nsd.subject):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def sigma_key(sigma: float) -> str:
    return f"sigma_{float(sigma):.3f}"


# ============================================================ 학습: 디코더

def train_decoders(cfg, names: Sequence[str] = DECODER_NAMES, log=print) -> None:
    mdir = models_dir(cfg)
    Xtr, (Xte,), st = zscore_fmri(cfg, load_fmri(cfg, "train"), [load_fmri(cfg, "test")])
    np.savez(mdir / "fmri_stats.npz", **st)
    dev = str(cfg.brain.solver_device)
    chunk = int(cfg.brain.chunk_size)
    report = {}
    for name in names:
        alpha = float(cfg.brain.decoders[name]["alpha"])
        Y = _flat(load_feature(cfg, "train", name))
        log(f"[brain] decoder {name}: fMRI {Xtr.shape} -> {Y.shape}, alpha={alpha:g}")
        lm = ridge.fit_ridge_fixed_alpha(Xtr, Y, alpha, mdir / f"dec_{name}", device=dev, chunk=chunk, log=log)
        Yte = _flat(load_feature(cfg, "test", name))
        pred = lm.predict(Xte, device=dev)
        r = ridge.pearson_columns(pred.astype(np.float64), np.asarray(Yte, dtype=np.float64))
        report[name] = dict(alpha=alpha, test_mean_r=float(np.nanmean(r)), test_median_r=float(np.nanmedian(r)))
        log(f"[brain]   test mean r = {report[name]['test_mean_r']:.4f}")
        del pred, Yte
    _write_json(mdir / "eval" / "decoders.json", report)


def train_control_text(cfg, log=print) -> None:
    """대조 사슬용: CLIP-V (cls + 풀링 패치) -> CLIP-T 를 뇌 없이 직접 잇는 릿지."""
    mdir = models_dir(cfg)
    dev = str(cfg.brain.solver_device)
    chunk = int(cfg.brain.chunk_size)
    Xtr = np.concatenate([np.asarray(load_feature(cfg, "train", "enc_clip_cls")),
                          np.asarray(load_feature(cfg, "train", "enc_clip_patch_pool"))], axis=1)
    Ytr = _flat(load_feature(cfg, "train", "cliptext"))
    mean = Xtr.mean(0)
    K = ridge.gram([(Xtr, mean, np.ones_like(mean), 1.0)], device=dev, chunk=chunk)
    rng = np.random.default_rng(int(cfg.project.seed))
    val = rng.choice(Xtr.shape[0], int(len(Xtr) * float(cfg.brain.encoder.val_fraction)), replace=False)
    alphas = [float(a) for a in cfg.brain.control_text_alphas]
    _, scores = ridge.select_alpha_holdout(K, np.asarray(Ytr, dtype=np.float32), alphas, val, log=log)
    best = alphas[int(np.nanargmax(np.nanmean(scores, axis=1)))]
    if best in (alphas[0], alphas[-1]):
        log(f"[brain] 경고: 대조 사상의 알파({best:g})가 격자 끝에 걸렸다. brain.control_text_alphas 를 넓힐 것")
    del K
    log(f"[brain] control text map alpha = {best:g}")
    lm = ridge.fit_ridge_fixed_alpha(Xtr, Ytr, best, mdir / "ctrl_cliptext", device=dev, chunk=chunk, log=log)
    Xte = np.concatenate([np.asarray(load_feature(cfg, "test", "enc_clip_cls")),
                          np.asarray(load_feature(cfg, "test", "enc_clip_patch_pool"))], axis=1)
    pred = lm.predict(Xte, device=dev)
    np.savez(_mk(mdir / "calib") / "control_text.npz",
             cliptext_mean=pred.mean(0), cliptext_std=pred.std(0).clip(1e-8), alpha=best)
    r = ridge.pearson_columns(pred.astype(np.float64), np.asarray(_flat(load_feature(cfg, "test", "cliptext")), dtype=np.float64))
    _write_json(mdir / "eval" / "control_text.json", dict(alpha=best, test_mean_r=float(np.nanmean(r))))


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================ 학습: 인코더

def _encoder_block_arrays(cfg, split: str) -> List[Tuple[str, np.ndarray]]:
    names = list(cfg.brain.encoder.blocks)
    file_of = {"clip_cls": "enc_clip_cls", "clip_patch_pool": "enc_clip_patch_pool", "vdvae": "vdvae"}
    return [(n, load_feature(cfg, split, file_of[n])) for n in names]


def train_encoder(cfg, log=print) -> None:
    mdir = models_dir(cfg)
    dev = str(cfg.brain.solver_device)
    chunk = int(cfg.brain.chunk_size)
    Ytr, (Yte,), st = zscore_fmri(cfg, load_fmri(cfg, "train"), [load_fmri(cfg, "test")])
    np.savez(mdir / "fmri_stats.npz", **st)

    arrays = _encoder_block_arrays(cfg, "train")
    blocks, meta = [], dict(names=[], dims=[], weights=[])
    means, scales = [], []
    for name, X in arrays:
        m = ridge.column_mean(X, chunk)
        s = ridge.column_std(X, m, chunk)
        w = 1.0 / X.shape[1]                       # 블록마다 그람 행렬 기여를 같게
        blocks.append((X, m, s, w))
        means.append(m)
        scales.append(s)
        meta["names"].append(name)
        meta["dims"].append(int(X.shape[1]))
        meta["weights"].append(w)
        log(f"[brain] encoder block {name}: {X.shape}")

    K = ridge.gram(blocks, device=dev, chunk=chunk)
    rng = np.random.default_rng(int(cfg.project.seed))
    n = Ytr.shape[0]
    val = np.sort(rng.choice(n, int(n * float(cfg.brain.encoder.val_fraction)), replace=False))
    alphas = [float(a) for a in cfg.brain.encoder.alphas]
    best_alpha, val_scores = ridge.select_alpha_holdout(K, Ytr, alphas, val, log=log)
    edge = float(np.mean((best_alpha == alphas[0]) | (best_alpha == alphas[-1])))
    if edge > 0.2:
        log(f"[brain] 경고: 복셀의 {edge:.0%} 가 알파 격자 끝을 골랐다. brain.encoder.alphas 를 넓힐 것")
    A, y_mean = ridge.dual_coef_per_target_alpha(K, Ytr, best_alpha)
    del K
    ridge.primal_from_dual(blocks, A, mdir / "encoder_W.npy", device=dev, chunk=chunk)
    np.savez(mdir / "encoder.npz", names=np.array(meta["names"]), dims=np.array(meta["dims"]),
             weights=np.array(meta["weights"]), mean=np.concatenate(means).astype(np.float32),
             scale=np.concatenate(scales).astype(np.float32), y_mean=y_mean.astype(np.float32),
             alpha=best_alpha, val_scores=val_scores, alphas=np.array(alphas))

    # 테스트셋 평가와 잔차(인공 뇌가 설명하지 못한 실제 뇌의 변동) -> 잡음 척도
    enc = Encoder.load(mdir)
    test_blocks = {name: np.asarray(X) for name, X in _encoder_block_arrays(cfg, "test")}
    pred = enc.predict(test_blocks, device=dev)
    r = ridge.pearson_columns(pred.astype(np.float64), Yte.astype(np.float64))
    resid_std = (Yte - pred).std(0, ddof=1).astype(np.float32)
    np.save(mdir / "noise_scale.npy", resid_std)
    np.save(_mk(mdir / "eval") / "encoder_test_r.npy", r)
    report = dict(test_median_r=float(np.nanmedian(r)), test_mean_r=float(np.nanmean(r)),
                  frac_r_gt_0p1=float(np.mean(r > 0.1)), frac_r_gt_0p3=float(np.mean(r > 0.3)),
                  alpha_counts={f"{a:g}": int((best_alpha == a).sum()) for a in alphas},
                  residual_std_median=float(np.median(resid_std)))
    _write_json(mdir / "eval" / "encoder.json", report)
    log(f"[brain] encoder test median r = {report['test_median_r']:.4f}")


# ============================================================ 실행 시점의 인공 뇌

def _as_tensor(a: np.ndarray, device: torch.device, dtype: torch.dtype, load: str) -> torch.Tensor:
    if load == "mmap" and device.type == "cpu" and dtype == torch.float32:
        return torch.from_numpy(np.asarray(a))      # memmap 을 그대로 씀 (페이지 캐시)
    return torch.from_numpy(np.array(a, dtype=np.float32)).to(device=device, dtype=dtype)   # RAM 으로 복사


class Encoder:
    def __init__(self, W, names, dims, weights, mean, scale, y_mean):
        self.W = W
        self.names = list(names)
        self.dims = [int(d) for d in dims]
        self.weights = [float(w) for w in weights]
        self.mean = mean
        self.scale = scale
        self.y_mean = y_mean

    @classmethod
    def load(cls, mdir: Path, mmap: bool = True) -> "Encoder":
        z = np.load(mdir / "encoder.npz", allow_pickle=True)
        W = np.load(mdir / "encoder_W.npy", mmap_mode="c" if mmap else None)
        return cls(W, z["names"], z["dims"], z["weights"], z["mean"], z["scale"], z["y_mean"])

    def standardize(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        parts, off = [], 0
        for name, d, w in zip(self.names, self.dims, self.weights):
            X = np.asarray(blocks[name], dtype=np.float32)
            parts.append((X - self.mean[off:off + d]) / self.scale[off:off + d] * np.float32(np.sqrt(w)))
            off += d
        return np.concatenate(parts, axis=1)

    def predict(self, blocks: Dict[str, np.ndarray], device: str = "cpu", chunk: int = 16384) -> np.ndarray:
        dev = ridge._dev(device)
        Z = torch.from_numpy(self.standardize(blocks)).to(dev)
        out = torch.zeros((Z.shape[0], self.W.shape[1]), dtype=torch.float32, device=dev)
        for a, b in ridge.column_chunks(self.W.shape[0], chunk):
            Wb = torch.from_numpy(np.asarray(self.W[a:b], dtype=np.float32)).to(dev)
            out += Z[:, a:b] @ Wb
        return out.cpu().numpy() + self.y_mean


class ArtificialBrain:
    """사슬에서 쓰는 인공 뇌. 행렬을 한 번 메모리(또는 GPU)에 올려 두고 그림 한 장씩 처리한다."""

    def __init__(self, cfg, backend, log=print):
        self.cfg = cfg
        self.backend = backend
        mdir = models_dir(cfg)
        self.mdir = mdir
        rt = cfg.brain.get("runtime", {}) if hasattr(cfg.brain, "get") else {}
        self.device = ridge._dev(str(rt.get("device", "cpu")))
        self.dtype = torch.float16 if str(rt.get("dtype", "float32")) == "float16" else torch.float32
        if self.device.type == "cpu" and self.dtype == torch.float16:
            log("[brain] CPU 에서는 float16 행렬곱을 쓸 수 없어 (torch 1.12) float32 로 바꾼다")
            self.dtype = torch.float32
        load = str(rt.get("load", "memory"))

        enc = Encoder.load(mdir)
        self.enc = enc
        self.W_E = _as_tensor(enc.W, self.device, self.dtype, load)
        self.noise_scale = np.load(mdir / "noise_scale.npy").astype(np.float32)
        self.dec: Dict[str, Tuple[torch.Tensor, np.ndarray, np.ndarray]] = {}
        for name in DECODER_NAMES:
            lm = ridge.LinearMap.load(mdir / f"dec_{name}")
            self.dec[name] = (_as_tensor(lm.W_T, self.device, self.dtype, load), lm.x_mean.astype(np.float32),
                              lm.y_mean.astype(np.float32))
            log(f"[brain] loaded decoder {name}: {tuple(lm.W_T.shape)}")
        ctrl = mdir / "ctrl_cliptext.W_T.npy"
        self.ctrl = None
        if ctrl.exists():
            lm = ridge.LinearMap.load(mdir / "ctrl_cliptext")
            self.ctrl = (_as_tensor(lm.W_T, self.device, self.dtype, load), lm.x_mean.astype(np.float32),
                         lm.y_mean.astype(np.float32))
        self.fstats = dict(np.load(feature_dir(cfg) / "train_feature_stats.npz"))
        self._calib: Dict[str, Dict[str, np.ndarray]] = {}
        self.shapes = dict(vdvae=(backend.vdvae_dim,),
                           clipvision=(backend.n_clip_tokens, backend.clip_dim),
                           cliptext=(backend.n_text_tokens, backend.clip_dim))

    # ---------------------------------------------------------------- 단계
    def features(self, images) -> Dict[str, np.ndarray]:
        """그림 -> 같은 모델의 특징 (vdvae 잠재, CLIP-V 토큰)."""
        return dict(vdvae=self.backend.vdvae_encode(images), clipvision=self.backend.clip_vision(images))

    def encode(self, feats: Dict[str, np.ndarray]) -> np.ndarray:
        blocks = encoder_blocks(feats["clipvision"], feats["vdvae"], self.enc.names,
                                int(self.cfg.brain.encoder.clip_patch_pool_grid))
        Z = torch.from_numpy(self.enc.standardize(blocks)).to(self.device, self.dtype)
        y = (Z @ self.W_E).float().cpu().numpy()
        return y + self.enc.y_mean

    def perturb(self, y: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        if sigma <= 0:
            return y
        scale = self.noise_scale if str(self.cfg.chain.noise_scale) == "residual" else np.float32(1.0)
        return (y + sigma * scale * rng.standard_normal(y.shape).astype(np.float32)).astype(np.float32)

    def _linear(self, mat: Tuple[torch.Tensor, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
        W_T, x_mean, y_mean = mat
        xc = torch.from_numpy((np.atleast_2d(x) - x_mean).astype(np.float32)).to(self.device, self.dtype)
        return (xc @ W_T.T).float().cpu().numpy() + y_mean

    def decode_raw(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: self._linear(self.dec[name], y) for name in DECODER_NAMES}

    def calib(self, sigma: float) -> Dict[str, np.ndarray]:
        key = sigma_key(sigma)
        if key not in self._calib:
            p = self.mdir / "calib" / f"{key}.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} 가 없다. scripts/07_calibrate.py 를 sigma={sigma} 로 먼저 실행할 것")
            self._calib[key] = dict(np.load(p))
        return self._calib[key]

    def renorm(self, raw: Dict[str, np.ndarray], sigma: float) -> Dict[str, np.ndarray]:
        mode = str(self.cfg.chain.renorm)
        out = {}
        cal = self.calib(sigma) if mode == "calibrated" else None
        for name, v in raw.items():
            if cal is not None:
                v = (v - cal[f"{name}_mean"]) / cal[f"{name}_std"] * self.fstats[f"{name}_std"] + self.fstats[f"{name}_mean"]
            out[name] = v.reshape((v.shape[0],) + self.shapes[name]).astype(np.float32)
        return out

    def control_text(self, clip_tokens: np.ndarray) -> Optional[np.ndarray]:
        if self.ctrl is None or str(self.cfg.chain.control_text) == "empty":
            return None
        grid = int(self.cfg.brain.encoder.clip_patch_pool_grid)
        b = encoder_blocks(clip_tokens, np.zeros((clip_tokens.shape[0], 1)), ["clip_cls", "clip_patch_pool"], grid)
        x = np.concatenate([b["clip_cls"], b["clip_patch_pool"]], axis=1)
        v = self._linear(self.ctrl, x)
        cal = np.load(self.mdir / "calib" / "control_text.npz")
        v = (v - cal["cliptext_mean"]) / cal["cliptext_std"] * self.fstats["cliptext_std"] + self.fstats["cliptext_mean"]
        return v.reshape((v.shape[0],) + self.shapes["cliptext"]).astype(np.float32)


# ============================================================ 보정

def calibrate(cfg, sigmas: Sequence[float], log=print) -> None:
    """NSD 테스트 그림의 특징 -> E -> (+잡음) -> D 의 출력 통계를 sigma 마다 저장한다."""
    mdir = models_dir(cfg)
    dev = str(cfg.brain.solver_device)
    enc = Encoder.load(mdir)
    blocks = {name: np.asarray(X) for name, X in _encoder_block_arrays(cfg, "test")}
    y = enc.predict(blocks, device=dev)
    noise_scale = np.load(mdir / "noise_scale.npy").astype(np.float32)
    rng = np.random.default_rng(int(cfg.project.seed) + 777)
    out_dir = _mk(mdir / "calib")
    for sigma in sigmas:
        scale = noise_scale if str(cfg.chain.noise_scale) == "residual" else np.float32(1.0)
        y_s = y + float(sigma) * scale * rng.standard_normal(y.shape).astype(np.float32) if sigma > 0 else y
        stats = {}
        for name in DECODER_NAMES:
            lm = ridge.LinearMap.load(mdir / f"dec_{name}")
            pred = lm.predict(y_s, device=dev)
            stats[f"{name}_mean"] = pred.mean(0).astype(np.float32)
            stats[f"{name}_std"] = pred.std(0).clip(1e-8).astype(np.float32)
            del pred
        np.savez(out_dir / f"{sigma_key(sigma)}.npz", **stats)
        log(f"[brain] calibration saved for sigma={sigma}")
