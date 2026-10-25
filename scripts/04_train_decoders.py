"""디코더 D 학습: fMRI -> VDVAE 잠재 / CLIP-V 토큰 / CLIP-T 토큰 (Brain-Diffuser 알파).
대조 사슬에 쓰는 CLIP-V -> CLIP-T 직접 사상도 함께 학습한다.

  python scripts/04_train_decoders.py --config configs/local.yaml
  python scripts/04_train_decoders.py --config configs/local.yaml --only clipvision

RAM 32GB 이상 권장. 가중치는 models/subj01/dec_*.W_T.npy 에 float32 로 저장된다 (대조용 포함 약 25GB).
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.brain import DECODER_NAMES, train_control_text, train_decoders
from seismo.config import add_config_args, config_from_args


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--only", nargs="+", default=list(DECODER_NAMES) + ["control_text"],
                   choices=list(DECODER_NAMES) + ["control_text"])
    args = p.parse_args()
    cfg = config_from_args(args)
    names = [n for n in args.only if n in DECODER_NAMES]
    if names:
        train_decoders(cfg, names=names, log=log)
    if "control_text" in args.only:
        train_control_text(cfg, log=log)


if __name__ == "__main__":
    main()
