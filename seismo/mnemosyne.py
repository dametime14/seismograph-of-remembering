"""2부 〈므네모시네〉 : 사슬 속에서 감정이 형태보다 오래 살아남는가.

  형태의 생존   CLIP 임베딩이 원본과 닮은 정도 (clip_sim_to_origin) 가 기준 아래로 처음 떨어지는 단계
  감정의 생존   감정 벡터 (34 범주, 라벨 평균을 뺀 것) 가 원본과 닮은 정도가 기준 아래로 처음 떨어지는 단계

Warburg 의 엔그램(살아남는 흔적)과 Bartlett 의 스키마(다시 지어지는 기억)를 같은 사슬에서 함께 잰다.
생존 길이로 패널 46 의 그림들을 다시 배열하고, 빈자리는 생성 모델이 패널의 평균 임베딩 근처에서
만든 후보 가운데 감정 점수가 가장 가까운 것으로 채운다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from PIL import Image

from .chain import iter_chain_dirs, read_log
from .metrics import cosine


def first_drop(sim: np.ndarray, threshold: float) -> int:
    """sim[t] (t>=1) 가 처음으로 threshold 아래가 되는 t. 끝까지 버티면 len(sim) (중도 절단)."""
    below = np.where(sim[1:] < threshold)[0]
    return int(below[0] + 1) if len(below) else int(len(sim))


def survival_table(cfg, emotion_model, out_root: Path, log=print) -> Path:
    m = cfg.mnemosyne
    rows = []
    traj = {}
    for d in iter_chain_dirs(out_root):
        spec = json.loads((d / "spec.json").read_text())
        recs = read_log(d)
        form = np.array([1.0] + [r[str(m.form_metric)] for r in recs])
        cls = np.load(d / "trace.npz")["clip_cls"]
        emo = emotion_model.centered(emotion_model.predict(cls))
        emo_sim = np.array([cosine(e, emo[0]) for e in emo])
        T = len(form) - 1
        fs = first_drop(form, float(m.form_threshold))
        es = first_drop(emo_sim, float(m.emotion_threshold))
        rows.append(dict(**spec, form_survival=fs, emotion_survival=es,
                         form_censored=int(fs > T), emotion_censored=int(es > T),
                         form_auc=float(form[1:].mean()), emotion_auc=float(emo_sim[1:].mean()),
                         emotion_minus_form=es - fs))
        traj[str(d.relative_to(out_root))] = np.stack([form, emo_sim])
    if not rows:
        raise FileNotFoundError(f"{out_root} 에 끝난 사슬이 없다")
    out = out_root / "survival.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(out_root / "survival_trajectories.npz", **{k.replace("/", "__"): v for k, v in traj.items()})
    log(f"[mnemosyne] {len(rows)} chains -> {out}")
    return out


def order_panel(cfg, survival_csv: Path, log=print) -> Dict:
    import pandas as pd
    panel = yaml.safe_load(open(cfg.get_path("mnemosyne.panel"), encoding="utf-8"))
    df = pd.read_csv(survival_csv)
    sel = df[(df.condition == panel.get("condition", "brain"))]
    if panel.get("condition", "brain") == "brain":
        sel = sel[np.isclose(sel.sigma, float(panel.get("sigma", 0.5)))]
    sel = sel[sel.artwork_id.astype(str).isin([str(i) for i in panel["images"]])]
    agg = (sel.groupby("artwork_id")[["emotion_survival", "form_survival", "emotion_auc", "form_auc"]]
           .mean().sort_values(["emotion_survival", "emotion_auc"], ascending=False))
    order = [str(i) for i in agg.index]
    result = dict(panel=panel.get("name", "panel"), order=order, table=agg.reset_index().to_dict(orient="records"))
    out = survival_csv.parent / f"panel_order_{panel.get('slug', 'panel')}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"[mnemosyne] panel order -> {out}")
    return result


def fill_vacancy(cfg, backend, emotion_model, image_paths: List[Path], out_dir: Path, log=print) -> List[Dict]:
    """패널 그림들의 평균 CLIP 토큰을 조건으로 순수 잡음에서 후보를 만들고 감정 점수로 고른다."""
    from .features import load_image
    v = cfg.mnemosyne.vacancy
    out_dir.mkdir(parents=True, exist_ok=True)
    ims = [load_image(p, size=int(cfg.chain.image_size)) for p in image_paths]
    tok = backend.clip_vision(ims)
    mean_tok = tok.mean(0, keepdims=True)
    mean_tok = mean_tok / max(np.linalg.norm(mean_tok[0, 0]), 1e-8)
    target = emotion_model.centered(emotion_model.predict(tok[:, 0])).mean(0)
    gray = Image.new("RGB", (512, 512), (127, 127, 127))
    scored = []
    for k in range(int(v.n_candidates)):
        img = backend.generate([gray], mean_tok, None, seed=int(cfg.project.seed) * 10_000 + k, strength=1.0)[0]
        emb = backend.clip_embed([img])
        e = emotion_model.centered(emotion_model.predict(emb))[0]
        scored.append(dict(k=k, emotion_cos=cosine(e, target), clip_cos=cosine(emb[0], mean_tok[0, 0]), img=img))
        if (k + 1) % 16 == 0:
            log(f"[mnemosyne] candidates {k + 1}/{v.n_candidates}")
    scored.sort(key=lambda r: r["emotion_cos"], reverse=True)
    keep = []
    for rank, r in enumerate(scored[: int(v.keep_top)]):
        name = f"vacancy_rank{rank:02d}_k{r['k']:04d}.png"
        r["img"].save(out_dir / name)
        keep.append(dict(file=name, k=r["k"], emotion_cos=r["emotion_cos"], clip_cos=r["clip_cos"]))
    (out_dir / "vacancy.json").write_text(json.dumps(keep, indent=2))
    return keep
