"""2부 〈므네모시네〉 : 형태와 감정의 생존 길이, 패널 재배열, 빈자리 채우기.

  python scripts/10_mnemosyne.py --config configs/local.yaml                 # 생존표 + 패널 순서
  python scripts/10_mnemosyne.py --config configs/local.yaml --vacancy       # 빈자리 후보 생성까지 (GPU)
"""
import argparse

import yaml

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.chain import list_artworks
from seismo.config import add_config_args, config_from_args
from seismo.emotion import EmotionModel
from seismo.mnemosyne import fill_vacancy, order_panel, survival_table


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--vacancy", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    emo = EmotionModel.load(cfg)
    out_root = cfg.get_path("paths.outputs") / "chains"
    csv_path = survival_table(cfg, emo, out_root, log=log)
    order_panel(cfg, csv_path, log=log)
    if args.vacancy:
        from pathlib import Path
        from seismo.backends import get_backend
        panel = yaml.safe_load(open(cfg.get_path("mnemosyne.panel"), encoding="utf-8"))
        arts = {a["id"]: a["path"] for a in list_artworks(cfg)}
        paths = [Path(arts[str(i)]) for i in panel["images"] if str(i) in arts]
        backend = get_backend(cfg, need=("vd",))
        out = cfg.get_path("paths.outputs") / "mnemosyne" / str(panel.get("slug", "panel"))
        keep = fill_vacancy(cfg, backend, emo, paths, out, log=log)
        log(f"[mnemosyne] vacancy candidates kept: {len(keep)} -> {out}")


if __name__ == "__main__":
    main()
