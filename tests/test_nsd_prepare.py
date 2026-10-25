"""NSD 폴더 구조를 흉내 낸 아주 작은 가짜 자료로 02 단계를 끝까지 돌려 본다."""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
nib = pytest.importorskip("nibabel")
spio = pytest.importorskip("scipy.io")

from seismo.config import load_config
from seismo.nsd import prepare_subject, split_trials


def _fake_nsd(root, n_sessions=2, shape=(4, 3, 2), n_images=40):
    rng = np.random.default_rng(0)
    exp = root / "nsddata/experiments/nsd"
    exp.mkdir(parents=True)
    n_trials = n_sessions * 750
    # 앞 1000 은 공유(테스트), 그 뒤는 학습. 여기서는 1..1010 중에서 고른다
    masterordering = np.concatenate([rng.integers(1, 1011, size=n_trials), np.ones(30000 - n_trials, dtype=int)])
    subjectim = np.tile(np.arange(1, 10001), (8, 1))
    spio.savemat(str(exp / "nsd_expdesign.mat"), dict(masterordering=masterordering, subjectim=subjectim))
    roi_dir = root / "nsddata/ppdata/subj01/func1pt8mm/roi"
    roi_dir.mkdir(parents=True)
    mask = np.zeros(shape)
    mask.flat[[1, 3, 5, 7, 9]] = 1
    mask.flat[0] = -1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(roi_dir / "nsdgeneral.nii.gz"))
    bdir = root / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    bdir.mkdir(parents=True)
    for s in range(n_sessions):
        data = (rng.standard_normal(shape + (750,)) * 300).astype(np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(bdir / f"betas_session{s + 1:02d}.nii.gz"))
    sdir = root / "nsddata_stimuli/stimuli/nsd"
    sdir.mkdir(parents=True)
    with h5py.File(str(sdir / "nsd_stimuli.hdf5"), "w") as f:
        f.create_dataset("imgBrick", data=rng.integers(0, 255, size=(1011, 8, 8, 3), dtype=np.uint8))
    annots = np.array([[f"cap{i}_{j}" for j in range(5)] for i in range(1011)])
    np.save(root / "COCO_73k_annots_curated.npy", annots)
    return masterordering, mask


def test_prepare_subject(tmp_path):
    nsd = tmp_path / "nsd"
    masterordering, mask = _fake_nsd(nsd)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "c.yaml").write_text(
        f"paths:\n  root: .\n  nsd_root: {nsd}\n  processed: {tmp_path}/processed\n"
        "nsd:\n  subject: 1\n  roi: nsdgeneral\n  n_sessions: 2\n  beta_version: betas_fithrf_GLMdenoise_RR\n"
        "  space: func1pt8mm\n  beta_scale: 300.0\n")
    cfg = load_config(cfg_dir / "c.yaml")
    prepare_subject(cfg, log=lambda *_: None)
    out = tmp_path / "processed/subj01"
    train, test = split_trials(nsd / "nsddata/experiments/nsd/nsd_expdesign.mat", 1, 2)
    tr_fmri = np.load(out / "train_fmri.npy")
    te_stim = np.load(out / "test_stim.npy")
    assert tr_fmri.shape == (len(train), 5)
    assert te_stim.shape == (len(test), 8, 8, 3) and te_stim.dtype == np.uint8
    # 그림 단위 평균이 맞는지 한 장 확인
    nsd_id = list(train.keys())[0]
    trials = sorted(train[nsd_id])
    bdir = nsd / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    betas = np.concatenate([
        np.asarray(nib.load(str(bdir / f"betas_session{s + 1:02d}.nii.gz")).dataobj)[mask > 0].T
        for s in range(2)])
    np.testing.assert_allclose(tr_fmri[0], betas[trials].mean(0), rtol=1e-5)
    caps = np.load(out / "train_cap.npy")
    assert caps[0, 0] == f"cap{nsd_id}_0"
