import numpy as np
import pytest

from seismo import ridge


def _primal(X, Y, alpha):
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, xm, ym


@pytest.mark.parametrize("n,p,q", [(40, 25, 7), (30, 60, 5)])
def test_fixed_alpha_matches_closed_form(tmp_path, n, p, q):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p)).astype(np.float32)
    Y = rng.standard_normal((n, q)).astype(np.float32)
    alpha = 3.0
    lm = ridge.fit_ridge_fixed_alpha(X, Y, alpha, tmp_path / "m", chunk=3, log=lambda *_: None)
    W, xm, ym = _primal(X.astype(np.float64), Y.astype(np.float64), alpha)
    Xt = rng.standard_normal((6, p)).astype(np.float32)
    np.testing.assert_allclose(lm.predict(Xt), (Xt - xm) @ W + ym, rtol=1e-3, atol=1e-4)
    lm2 = ridge.LinearMap.load(tmp_path / "m")
    np.testing.assert_allclose(lm2.predict(Xt), lm.predict(Xt), rtol=1e-6)


def test_matches_sklearn(tmp_path):
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 80)).astype(np.float32)
    Y = rng.standard_normal((50, 4)).astype(np.float32)
    lm = ridge.fit_ridge_fixed_alpha(X, Y, 10.0, tmp_path / "m", log=lambda *_: None)
    ref = sk.Ridge(alpha=10.0, fit_intercept=True).fit(X, Y)
    Xt = rng.standard_normal((5, 80)).astype(np.float32)
    np.testing.assert_allclose(lm.predict(Xt), ref.predict(Xt), rtol=1e-3, atol=1e-4)


def test_dual_per_target_alpha_and_primal(tmp_path):
    rng = np.random.default_rng(2)
    n = 45
    X1 = rng.standard_normal((n, 12)).astype(np.float32)
    X2 = rng.standard_normal((n, 30)).astype(np.float32) * 5 + 2
    Y = rng.standard_normal((n, 6)).astype(np.float32)
    blocks = []
    for X in (X1, X2):
        m = ridge.column_mean(X)
        s = ridge.column_std(X, m)
        blocks.append((X, m, s, 1.0 / X.shape[1]))
    K = ridge.gram(blocks)
    alphas = np.array([0.1, 1.0, 10.0, 0.1, 1.0, 10.0])
    A, y_mean = ridge.dual_coef_per_target_alpha(K, Y, alphas)
    W = ridge.primal_from_dual(blocks, A, tmp_path / "W.npy")
    Z = np.concatenate([((X - m) / s) * np.sqrt(w) for X, m, s, w in blocks], axis=1)
    for j, a in enumerate(alphas):
        Wj, _, ym = _primal(Z.astype(np.float64), Y[:, j:j + 1].astype(np.float64), a)
        np.testing.assert_allclose(Z @ W[:, j:j + 1] + y_mean[j], Z @ Wj + ym, rtol=1e-3, atol=1e-4)


def test_select_alpha_prefers_signal():
    rng = np.random.default_rng(3)
    n, p = 200, 10
    X = rng.standard_normal((n, p)).astype(np.float32)
    # 앞 두 열은 X 의 선형 함수, 뒤 두 열은 순수 잡음
    Y = np.concatenate([X @ rng.standard_normal((p, 2)), rng.standard_normal((n, 2))], axis=1)
    m = X.mean(0)
    K = ridge.gram([(X, m, np.ones(p), 1.0)])
    best, scores = ridge.select_alpha_holdout(K, Y.astype(np.float32), [1e-2, 1e2, 1e5],
                                              np.arange(160, 200), log=lambda *_: None)
    assert scores.shape == (3, 4)
    assert scores[:, :2].max() > 0.9
