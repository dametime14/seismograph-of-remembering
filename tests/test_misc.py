import numpy as np

from seismo.config import load_config
from seismo.features import clip_patch_pool, pool_grid
from seismo.mnemosyne import first_drop


def test_config_interpolation_and_override(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "c.yaml").write_text(
        "paths:\n  root: .\n  data_root: ./data\n  nsd_root: ${paths.data_root}/nsd\nchain:\n  n_iters: 20\n")
    cfg = load_config(cfg_dir / "c.yaml", ["chain.n_iters=3", "paths.data_root=/x"])
    assert cfg.chain.n_iters == 3
    assert cfg.paths.nsd_root == "/x/nsd"
    assert str(cfg.get_path("paths.nsd_root")) == "/x/nsd"


def test_default_config_loads():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.nsd.subject == 1
    assert "${" not in cfg.paths.processed
    assert cfg.brain.decoders.clipvision.alpha == 60000


def test_clip_patch_pool():
    tok = np.zeros((2, 1 + 16 * 16, 3), dtype=np.float32)
    tok[:, 1:, 0] = np.arange(256)
    pooled = clip_patch_pool(tok, 4)
    assert pooled.shape == (2, 16 * 3)
    grid = tok[0, 1:, 0].reshape(16, 16)
    assert np.isclose(pooled[0, 0], grid[:4, :4].mean())
    assert pool_grid(16, 4) == 4 and pool_grid(2, 4) == 2 and pool_grid(6, 4) == 3


def test_first_drop():
    sim = np.array([1.0, 0.9, 0.8, 0.6, 0.7])
    assert first_drop(sim, 0.75) == 3
    assert first_drop(sim, 0.5) == len(sim)      # 끝까지 살아남음 (중도 절단)
