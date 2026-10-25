"""NSD 준비: 시행(trial) 단위 beta 를 그림 단위로 평균해 학습/테스트 배열을 만든다.

Brain-Diffuser 의 data/prepare_nsddata.py 와 같은 분할을 따른다.
  공유 1000장(masterordering <= 1000) -> 테스트 (37세션 기준 982장)
  나머지 -> 학습 (subj01 기준 8,859장)
차이점: 자극 그림을 float64 가 아닌 uint8 로 저장하고, 73,000장 전체를 메모리에 올리지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def split_trials(expdesign_path: Path, subject: int, n_sessions: int) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    import scipy.io as spio

    d = spio.loadmat(str(expdesign_path), squeeze_me=True)
    masterordering = np.asarray(d["masterordering"]).astype(int)   # 1-based, 길이 30000
    subjectim = np.asarray(d["subjectim"]).astype(int)             # (8, 10000), 1-based nsdId
    n_trials = n_sessions * 750
    train: Dict[int, List[int]] = {}
    test: Dict[int, List[int]] = {}
    for idx in range(n_trials):
        mo = masterordering[idx]
        nsd_id = subjectim[subject - 1, mo - 1] - 1                # 0-based nsdId
        target = train if mo > 1000 else test
        target.setdefault(int(nsd_id), []).append(idx)
    return train, test


def load_masked_betas(betas_dir: Path, mask: np.ndarray, n_sessions: int, log=print) -> np.ndarray:
    import nibabel as nib

    n_vox = int((mask > 0).sum())
    fmri = np.zeros((n_sessions * 750, n_vox), dtype=np.float32)
    sel = mask > 0
    for s in range(n_sessions):
        f = betas_dir / f"betas_session{s + 1:02d}.nii.gz"
        img = nib.load(str(f))
        data = np.asanyarray(img.dataobj)                          # int16 그대로, 메모리 절약
        fmri[s * 750:(s + 1) * 750] = data[sel].T.astype(np.float32)
        del data, img
        log(f"[nsd] session {s + 1}/{n_sessions} loaded")
    return fmri


def prepare_subject(cfg, log=print) -> None:
    import h5py
    import nibabel as nib

    sub = int(cfg.nsd.subject)
    root = cfg.get_path("paths.nsd_root")
    out = cfg.get_path("paths.processed") / f"subj{sub:02d}"
    out.mkdir(parents=True, exist_ok=True)

    train, test = split_trials(root / "nsddata/experiments/nsd/nsd_expdesign.mat", sub, int(cfg.nsd.n_sessions))
    log(f"[nsd] subj{sub:02d}: train {len(train)} images, test {len(test)} images")

    roi = nib.load(str(root / f"nsddata/ppdata/subj{sub:02d}/{cfg.nsd.space}/roi/{cfg.nsd.roi}.nii.gz")).get_fdata()
    betas_dir = root / f"nsddata_betas/ppdata/subj{sub:02d}/{cfg.nsd.space}/{cfg.nsd.beta_version}"
    fmri = load_masked_betas(betas_dir, roi, int(cfg.nsd.n_sessions), log)
    np.save(out / "roi_mask.npy", (roi > 0))

    annots_path = root / "COCO_73k_annots_curated.npy"
    annots = np.load(annots_path, allow_pickle=True) if annots_path.exists() else None
    if annots is None:
        log(f"[nsd] 경고: {annots_path} 가 없어 캡션을 저장하지 않는다 (CLIP-Text 디코더 학습에 필요)")

    stim_h5 = h5py.File(str(root / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"), "r")
    brick = stim_h5["imgBrick"]

    for name, table in (("train", train), ("test", test)):
        ids = list(table.keys())                                  # Brain-Diffuser 와 같은 순서 (등장 순)
        n = len(ids)
        f_avg = np.zeros((n, fmri.shape[1]), dtype=np.float32)
        n_rep = np.zeros(n, dtype=np.int16)
        for i, nsd_id in enumerate(ids):
            trials = sorted(table[nsd_id])
            f_avg[i] = fmri[trials].mean(0)
            n_rep[i] = len(trials)
        np.save(out / f"{name}_fmri.npy", f_avg)                  # 원 단위 (beta * 300). 표준화는 학습 때.
        np.save(out / f"{name}_nsd_ids.npy", np.asarray(ids, dtype=np.int64))
        np.save(out / f"{name}_n_repeats.npy", n_rep)

        stim = np.lib.format.open_memmap(out / f"{name}_stim.npy", mode="w+", dtype=np.uint8,
                                         shape=(n, brick.shape[1], brick.shape[2], brick.shape[3]))
        for i, nsd_id in enumerate(ids):
            stim[i] = brick[nsd_id]
            if (i + 1) % 1000 == 0:
                log(f"[nsd] {name} stimuli {i + 1}/{n}")
        stim.flush()
        del stim

        if annots is not None:
            caps = np.asarray(annots[ids])
            np.save(out / f"{name}_cap.npy", caps)
        log(f"[nsd] {name} saved ({n} images)")

    stim_h5.close()
    meta = dict(subject=sub, n_sessions=int(cfg.nsd.n_sessions), roi=cfg.nsd.roi,
                n_voxels=int(fmri.shape[1]), n_train=len(train), n_test=len(test))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[nsd] done: {meta}")


def load_fmri(cfg, split: str) -> np.ndarray:
    sub = int(cfg.nsd.subject)
    return np.load(cfg.get_path("paths.processed") / f"subj{sub:02d}" / f"{split}_fmri.npy").astype(np.float32)


def zscore_fmri(cfg, train: np.ndarray, others: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
    """Brain-Diffuser 와 동일: /300 후 학습셋 평균/표준편차(ddof=1)로 표준화."""
    scale = float(cfg.nsd.beta_scale)
    train = train / scale
    mean = train.mean(0)
    std = train.std(0, ddof=1)
    std[std == 0] = 1.0
    z_train = ((train - mean) / std).astype(np.float32)
    z_others = [(((o / scale) - mean) / std).astype(np.float32) for o in others]
    return z_train, z_others, dict(mean=mean, std=std, scale=np.float64(scale))
