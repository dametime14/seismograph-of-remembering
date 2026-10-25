"""감정 모델: CLIP 이미지 임베딩 -> Cowen & Keltner 34 범주 점수.

Horikawa et al. (2020) 가 fMRI 실험에 쓴 Cowen & Keltner (2017) 영상 2,185개와 그 34 범주 평정을 쓴다.
라벨 표의 기본 형식은 사용자의 EmoBrain 저장소의 cowen_horikawa_labels.csv 를 따른다
(정수 자극 번호 stim_num_int, score_0 .. score_33, 영상 파일 0001.mp4 ..).

단일 모델 원칙에 따라 특징은 사슬과 같은 CLIP ViT-L/14 (VD 안의 CLIP) 의 CLS 임베딩이다.
영상에서 프레임 몇 장을 고르게 뽑아 임베딩을 평균한다. 정지 그림(회화)에 그대로 적용할 때의
영역 차이는 열린 질문으로 남아 있다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def read_frames(video_path: Path, k: int) -> List[Image.Image]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    idx = np.linspace(0, n - 1, num=min(k, n)).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def load_labels(cfg):
    import pandas as pd
    e = cfg.emotion
    df = pd.read_csv(str(cfg.get_path("emotion.labels_csv")))
    cols = [f"{e.score_columns_prefix}{i}" for i in range(int(e.n_categories))]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"라벨 표에 없는 열: {missing[:5]} ...")
    ids = df[e.id_column].astype(int).to_numpy()
    return ids, df[cols].to_numpy(dtype=np.float32)


def embed_videos(cfg, backend, log=print) -> Path:
    e = cfg.emotion
    root = cfg.get_path("paths.emotion_root")
    out = root / "clip_video_embeddings.npz"
    ids, _ = load_labels(cfg)
    vdir = cfg.get_path("emotion.videos_dir")
    emb = np.zeros((len(ids), backend.clip_dim), dtype=np.float32)
    ok = np.zeros(len(ids), dtype=bool)
    for i, sid in enumerate(ids):
        frames = read_frames(vdir / str(e.video_name_format).format(int(sid)), int(e.frames_per_video))
        if not frames:
            continue
        v = backend.clip_embed(frames).mean(0)
        emb[i] = v / max(np.linalg.norm(v), 1e-8)
        ok[i] = True
        if (i + 1) % 100 == 0:
            log(f"[emotion] embedded {i + 1}/{len(ids)}")
    np.savez(out, ids=ids, emb=emb, ok=ok)
    log(f"[emotion] {ok.sum()}/{len(ids)} videos embedded -> {out}")
    return out


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float):
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, xm, ym


def _corr_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    den[den == 0] = np.inf
    return (a * b).sum(0) / den


def train_emotion_model(cfg, log=print) -> Dict:
    root = cfg.get_path("paths.emotion_root")
    z = np.load(root / "clip_video_embeddings.npz")
    ids_lab, Y = load_labels(cfg)
    assert np.array_equal(z["ids"], ids_lab), "임베딩과 라벨의 자극 순서가 다르다"
    ok = z["ok"]
    X, Y = z["emb"][ok].astype(np.float64), Y[ok].astype(np.float64)
    rng = np.random.default_rng(int(cfg.project.seed))
    perm = rng.permutation(len(X))
    n_te = n_va = max(1, len(X) // 10)
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    best, best_r = None, -np.inf
    for alpha in cfg.emotion.alphas:
        W, xm, ym = _ridge_fit(X[tr], Y[tr], float(alpha))
        r = np.nanmean(_corr_cols((X[va] - xm) @ W + ym, Y[va]))
        log(f"[emotion] alpha={float(alpha):g} val mean r={r:.4f}")
        if r > best_r:
            best, best_r = float(alpha), r
    W, xm, ym = _ridge_fit(X[np.concatenate([tr, va])], Y[np.concatenate([tr, va])], best)
    r_te = _corr_cols((X[te] - xm) @ W + ym, Y[te])
    np.savez(root / "emotion_model.npz", W=W.astype(np.float32), x_mean=xm.astype(np.float32),
             y_mean=ym.astype(np.float32), label_mean=Y.mean(0).astype(np.float32), alpha=best,
             test_r=r_te.astype(np.float32))
    names = list(cfg.emotion.category_names) or [f"score_{i}" for i in range(Y.shape[1])]
    report = dict(alpha=best, test_mean_r=float(np.nanmean(r_te)),
                  per_category_r={n: float(v) for n, v in zip(names, r_te)})
    (root / "emotion_model_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"[emotion] test mean r = {report['test_mean_r']:.4f}")
    return report


class EmotionModel:
    def __init__(self, W, x_mean, y_mean, label_mean):
        self.W, self.x_mean, self.y_mean, self.label_mean = W, x_mean, y_mean, label_mean

    @classmethod
    def load(cls, cfg) -> "EmotionModel":
        z = np.load(cfg.get_path("paths.emotion_root") / "emotion_model.npz")
        return cls(z["W"], z["x_mean"], z["y_mean"], z["label_mean"])

    def predict(self, cls_emb: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(cls_emb).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
        return (X - self.x_mean) @ self.W + self.y_mean

    def centered(self, scores: np.ndarray) -> np.ndarray:
        """비교용: 모든 비율 벡터가 비슷해 보이지 않도록 라벨 평균을 뺀다."""
        return scores - self.label_mean
