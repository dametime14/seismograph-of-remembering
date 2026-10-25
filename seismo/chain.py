"""1부 〈뇌를 지나간 그림〉 : Bartlett 의 연쇄 재생산을 인공 뇌 위에서 돌린다.

  뇌 사슬    x(t+1) = G( D( E(x(t)) + eps ) )      eps ~ N(0, (sigma * s_v)^2)
  대조 사슬  x(t+1) = G( F(x(t)) )                  F: 뇌를 거치지 않은 같은 모델의 특징

G 는 Brain-Diffuser 의 2단계 생성 (VDVAE 복원 -> Versatile Diffusion), E 는 인코더, D 는 디코더.
대조 사슬은 '생성 모델의 사전지식만으로 생기는 표류'를 보여 주고, 둘의 차이가 '뇌라는 병목'의 몫이다.
"""
from __future__ import annotations

import csv
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image

from .features import load_image
from .metrics import step_metrics


@dataclass(frozen=True)
class ChainSpec:
    artwork_id: str
    condition: str          # "brain" | "control"
    sigma: float
    replicate: int

    @property
    def tag(self) -> str:
        if self.condition == "control":
            return f"control/rep{self.replicate:02d}"
        return f"brain_sigma{self.sigma:.3f}/rep{self.replicate:02d}"

    def seed(self, base: int) -> int:
        key = f"{self.artwork_id}|{self.condition}|{self.sigma:.4f}|{self.replicate}"
        return int(base) * 1_000_003 + (zlib.crc32(key.encode()) % 1_000_000) * 1000


# ---------------------------------------------------------------- 그림 목록

def list_artworks(cfg) -> List[Dict[str, str]]:
    root = cfg.get_path("paths.artworks")
    table = root / "artworks.csv"
    rows: List[Dict[str, str]] = []
    if table.exists():
        with open(table, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["path"] = str(root / row["file"])
                rows.append(row)
    else:
        for p in sorted(root.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
                rows.append(dict(id=p.stem, file=p.name, path=str(p)))
    if not rows:
        raise FileNotFoundError(f"{root} 에 그림이 없다. configs/artworks.example.csv 를 참고할 것")
    return rows


def all_specs(cfg, artworks: List[Dict[str, str]]) -> List[ChainSpec]:
    specs = []
    for art in artworks:
        for rep in range(int(cfg.chain.n_replicates)):
            for sigma in cfg.chain.sigmas:
                specs.append(ChainSpec(art["id"], "brain", float(sigma), rep))
            if bool(cfg.chain.control):
                specs.append(ChainSpec(art["id"], "control", 0.0, rep))
    return specs


def shard(specs: List[ChainSpec], index: int, total: int) -> List[ChainSpec]:
    return [s for i, s in enumerate(specs) if i % total == index]


# ---------------------------------------------------------------- 한 단계

def brain_step(brain, backend, feats: Dict[str, np.ndarray], sigma: float, seed: int):
    rng = np.random.default_rng(seed)
    y = brain.encode(feats)
    y_noisy = brain.perturb(y, sigma, rng)
    cond = brain.renorm(brain.decode_raw(y_noisy), sigma)
    low = backend.vdvae_decode(cond["vdvae"], seed=seed)
    out = backend.generate(low, cond["clipvision"], cond["cliptext"], seed=seed)
    return out[0], low[0]


def control_step(brain, backend, feats: Dict[str, np.ndarray], seed: int):
    text = brain.control_text(feats["clipvision"]) if brain is not None else None
    low = backend.vdvae_decode(feats["vdvae"], seed=seed)
    out = backend.generate(low, feats["clipvision"], text, seed=seed)
    return out[0], low[0]


# ---------------------------------------------------------------- 사슬 하나

def run_chain(cfg, spec: ChainSpec, artwork_path: str, brain, backend, out_root: Path, log=print) -> Path:
    out = out_root / spec.artwork_id / spec.tag
    if (out / "trace.npz").exists():
        log(f"[chain] skip (done): {spec}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(asdict(spec), indent=2))

    size = int(cfg.chain.image_size)
    x0 = load_image(Path(artwork_path), size=size, mode=str(cfg.chain.get("crop", "center_crop")))
    x0.save(out / "t000.png")
    base_seed = spec.seed(int(cfg.project.seed))

    def feats_of(img: Image.Image) -> Dict[str, np.ndarray]:
        return dict(vdvae=backend.vdvae_encode([img]), clipvision=backend.clip_vision([img]))

    feats = feats_of(x0)
    cls0 = feats["clipvision"][0, 0]
    cls_list = [cls0]
    vox_list = [brain.encode(feats)[0]] if brain is not None else []
    prev, prev_cls = x0, cls0
    n_iters = int(cfg.chain.n_iters)
    save_every = int(cfg.chain.save_every)

    with open(out / "log.jsonl", "w", encoding="utf-8") as logf:
        for t in range(1, n_iters + 1):
            seed = base_seed + t
            if spec.condition == "brain":
                img, low = brain_step(brain, backend, feats, spec.sigma, seed)
            else:
                img, low = control_step(brain, backend, feats, seed)
            feats = feats_of(img)
            cls = feats["clipvision"][0, 0]
            rec = dict(t=t, seed=seed, **step_metrics(img, x0, prev, cls, cls0, prev_cls))
            if brain is not None:
                v = brain.encode(feats)[0]
                rec["voxel_corr_to_origin"] = float(np.corrcoef(v, vox_list[0])[0, 1])
                vox_list.append(v)
            cls_list.append(cls)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if t % save_every == 0 or t == n_iters:
                img.save(out / f"t{t:03d}.png")
                low.resize((128, 128), resample=Image.BICUBIC).save(out / f"t{t:03d}_low.png")
            log(f"[chain] {spec.tag} {spec.artwork_id} t={t} clip->origin={rec['clip_sim_to_origin']:.3f}")
            prev, prev_cls = img, cls

    trace = dict(clip_cls=np.stack(cls_list).astype(np.float32))
    if vox_list:
        trace["voxels"] = np.stack(vox_list).astype(np.float16)
    np.savez(out / "trace.npz", **trace)
    return out


def read_log(chain_dir: Path) -> List[Dict]:
    with open(chain_dir / "log.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_chain_dirs(out_root: Path) -> Iterable[Path]:
    for spec_file in sorted(out_root.glob("*/*/rep*/spec.json")):
        if (spec_file.parent / "trace.npz").exists():
            yield spec_file.parent
