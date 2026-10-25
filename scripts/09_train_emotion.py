"""감정 모델: Cowen & Keltner 영상(Horikawa 2020 자극)의 CLIP 임베딩 -> 34 범주 점수.

  python scripts/09_train_emotion.py --config configs/local.yaml
  python scripts/09_train_emotion.py --config configs/local.yaml --skip-embed   # 임베딩이 이미 있을 때

라벨 표(emotion.labels_csv)와 영상 폴더(emotion.videos_dir)를 설정에서 가리킬 것.
NERSC 에 이미 있는 EmoBrain 자료를 쓰려면 로컬 설정에서 paths.emotion_root 등을 그쪽으로 두면 된다.
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.config import add_config_args, config_from_args
from seismo.emotion import embed_videos, train_emotion_model


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--skip-embed", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    if not args.skip_embed:
        from seismo.backends import get_backend
        backend = get_backend(cfg, need=("clip",))
        embed_videos(cfg, backend, log=log)
    train_emotion_model(cfg, log=log)


if __name__ == "__main__":
    main()
