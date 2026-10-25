"""1부 〈뇌를 지나간 그림〉 : 그림마다 뇌 사슬(sigma 별)과 대조 사슬을 돌린다 (GPU).

  python scripts/08_run_chains.py --config configs/local.yaml
  python scripts/08_run_chains.py --config configs/local.yaml --artworks bruegel_icarus --set chain.n_iters=5
  # GPU 여러 장 또는 Slurm 배열 작업으로 나눌 때
  python scripts/08_run_chains.py --config configs/local.yaml --shard 0 --num-shards 4

결과: outputs/chains/<그림>/<조건>/repXX/{t000.png.., log.jsonl, trace.npz}
"""
import argparse

import _bootstrap  # noqa: F401
from _bootstrap import log
from seismo.backends import get_backend
from seismo.brain import ArtificialBrain
from seismo.chain import all_specs, list_artworks, run_chain, shard
from seismo.config import add_config_args, config_from_args
from seismo.viz import contact_sheet, similarity_curves


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--artworks", nargs="*", default=None, help="이 id 들만 돌린다")
    p.add_argument("--conditions", nargs="*", default=["brain", "control"], choices=["brain", "control"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    cfg = config_from_args(args)

    arts = list_artworks(cfg)
    if args.artworks:
        arts = [a for a in arts if a["id"] in set(args.artworks)]
    path_of = {a["id"]: a["path"] for a in arts}
    specs = [s for s in all_specs(cfg, arts) if s.condition in args.conditions]
    specs = shard(specs, args.shard, args.num_shards)
    log(f"[run] {len(arts)} artworks, {len(specs)} chains in this shard")

    backend = get_backend(cfg, need=("vdvae", "vd"))
    brain = ArtificialBrain(cfg, backend, log=log)
    out_root = cfg.get_path("paths.outputs") / "chains"
    for i, spec in enumerate(specs):
        log(f"[run] chain {i + 1}/{len(specs)}: {spec}")
        run_chain(cfg, spec, path_of[spec.artwork_id], brain, backend, out_root, log=log)

    n = int(cfg.chain.n_iters)
    steps = sorted({0, 1, 2, 3, 5, 8, 12, n // 2, n} & set(range(n + 1)))
    for art in arts:
        d = out_root / art["id"]
        if d.exists():
            contact_sheet(d, steps, d / "contact_sheet.png")
            similarity_curves(d, d / "clip_sim_to_origin.png")
    log("[run] done")


if __name__ == "__main__":
    main()
