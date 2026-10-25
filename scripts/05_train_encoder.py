"""인코더 E 학습: [CLIP cls, CLIP 풀링 패치, VDVAE 잠재] -> nsdgeneral 복셀 (복셀별 알파).

  python scripts/05_train_encoder.py --config configs/local.yaml

결과: models/subj01/encoder_W.npy, encoder.npz, noise_scale.npy, eval/encoder.json
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.brain import train_encoder
from seismo.config import add_config_args, config_from_args


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    train_encoder(cfg, log=log)


if __name__ == "__main__":
    main()
