"""점검: 실제 NSD 테스트 fMRI 로 Brain-Diffuser 재구성을 재현한다 (사슬을 돌리기 전에 꼭).

디코더 출력은 원 논문처럼 테스트 예측 전체의 배치 통계로 재정규화한다.
이 결과가 Brain-Diffuser 논문 그림과 비슷해야 이후 사슬을 믿을 수 있다.

  python scripts/06_sanity_reconstruct.py --config configs/local.yaml --n 12
"""
import argparse

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo import ridge
from seismo.backends import get_backend
from seismo.brain import DECODER_NAMES, models_dir
from seismo.config import add_config_args, config_from_args
from seismo.features import feature_dir, processed_dir
from seismo.nsd import load_fmri, zscore_fmri


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--n", type=int, default=12, help="재구성할 테스트 그림 수")
    args = p.parse_args()
    cfg = config_from_args(args)
    mdir = models_dir(cfg)
    _, (Xte,), _ = zscore_fmri(cfg, load_fmri(cfg, "train"), [load_fmri(cfg, "test")])
    fst = dict(np.load(feature_dir(cfg) / "train_feature_stats.npz"))
    backend = get_backend(cfg, need=("vdvae", "vd"))
    shapes = dict(vdvae=(backend.vdvae_dim,), clipvision=(backend.n_clip_tokens, backend.clip_dim),
                  cliptext=(backend.n_text_tokens, backend.clip_dim))
    cond = {}
    for name in DECODER_NAMES:
        pred = ridge.LinearMap.load(mdir / f"dec_{name}").predict(Xte, device=str(cfg.brain.solver_device))
        pred = (pred - pred.mean(0)) / pred.std(0).clip(1e-8) * fst[f"{name}_std"] + fst[f"{name}_mean"]
        cond[name] = pred[: args.n].reshape((args.n,) + shapes[name])
        del pred
    stim = np.load(processed_dir(cfg) / "test_stim.npy", mmap_mode="r")
    out = cfg.get_path("paths.outputs") / "sanity"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in range(args.n):
        low = backend.vdvae_decode(cond["vdvae"][i:i + 1], seed=i)
        img = backend.generate(low, cond["clipvision"][i:i + 1], cond["cliptext"][i:i + 1], seed=i)[0]
        orig = Image.fromarray(np.asarray(stim[i])).resize((256, 256))
        row = Image.new("RGB", (768, 256))
        row.paste(orig, (0, 0))
        row.paste(low[0].resize((256, 256)), (256, 0))
        row.paste(img.resize((256, 256)), (512, 0))
        row.save(out / f"test{i:03d}_orig_vdvae_vd.png")
        tiles.append(row)
        log(f"[sanity] {i + 1}/{args.n}")
    sheet = Image.new("RGB", (768, 256 * len(tiles)))
    for i, t in enumerate(tiles):
        sheet.paste(t, (0, 256 * i))
    sheet.save(out / "sanity_sheet.png")
    log(f"[sanity] saved -> {out} (열: 원본 | VDVAE 1단계 | VD 최종)")


if __name__ == "__main__":
    main()
