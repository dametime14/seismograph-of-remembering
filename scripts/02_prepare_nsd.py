"""NSD 원본 -> subj01 학습/테스트 배열 (그림 단위 평균 beta, 자극 uint8, 캡션).

  python scripts/02_prepare_nsd.py --config configs/local.yaml
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.config import add_config_args, config_from_args
from seismo.nsd import prepare_subject


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    prepare_subject(cfg, log=log)


if __name__ == "__main__":
    main()
