"""NSD 자극에서 VDVAE / CLIP-Vision / CLIP-Text 특징을 뽑는다 (GPU).

  python scripts/03_extract_features.py --config configs/local.yaml                 # 셋 다
  python scripts/03_extract_features.py --config configs/local.yaml --which vdvae   # 하나만
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.backends import get_backend
from seismo.config import add_config_args, config_from_args
from seismo.features import extract_nsd_features, feature_dir, feature_stats


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--which", nargs="+", default=["vdvae", "clipvision", "cliptext"],
                   choices=["vdvae", "clipvision", "cliptext"])
    p.add_argument("--stats-only", action="store_true", help="특징은 있고 통계만 다시 계산")
    args = p.parse_args()
    cfg = config_from_args(args)
    if not args.stats_only:
        need = []
        if "vdvae" in args.which:
            need.append("vdvae")
        if "clipvision" in args.which or "cliptext" in args.which:
            need.append("clip")
        backend = get_backend(cfg, need=need)
        extract_nsd_features(cfg, backend, args.which, log=log)
    # 이미 만들어진 특징 파일만 통계를 낸다 (--which 로 하나씩 돌릴 때도 앞서 만든 것의 통계를 함께 다시 씀)
    names = [n for n in ("vdvae", "clipvision", "cliptext") if (feature_dir(cfg) / f"train_{n}.npy").exists()]
    feature_stats(cfg, names=names, log=log)


if __name__ == "__main__":
    main()
