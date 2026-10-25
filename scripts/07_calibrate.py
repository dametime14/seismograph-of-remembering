"""보정 통계: NSD 테스트 982장의 특징 -> E -> (+잡음 sigma) -> D 출력의 차원별 평균/표준편차.

  python scripts/07_calibrate.py --config configs/local.yaml                  # chain.sigmas 전부
  python scripts/07_calibrate.py --config configs/local.yaml --sigmas 0 2.0   # 따로 지정
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.brain import calibrate
from seismo.config import add_config_args, config_from_args


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--sigmas", nargs="*", type=float, default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    sigmas = args.sigmas if args.sigmas else [float(s) for s in cfg.chain.sigmas]
    calibrate(cfg, sigmas, log=log)


if __name__ == "__main__":
    main()
