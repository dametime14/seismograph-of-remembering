"""Brain-Diffuser 백엔드 (VDVAE + CLIP ViT-L/14 + Versatile Diffusion).

원본 코드(ozcelikfu/brain-diffuser, MIT)를 third_party/brain-diffuser 에 그대로 두고 불러 쓴다.
원본 스크립트의 처리 순서와 하이퍼파라미터를 그대로 옮겼다:
  scripts/vdvae_extract_features.py, vdvae_reconstruct_images.py,
  scripts/clipvision_extract_features.py, cliptext_extract_features.py,
  scripts/versatilediffusion_reconstruct_images.py

원본은 GPU 두 장(12GB)을 가정했지만 여기서는 한 장(A100 40GB 이상 권장)에 모두 올린다.
다른 GPU 를 쓰려면 CUDA_VISIBLE_DEVICES 로 고르고 device 는 cuda:0 으로 둔다.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base import Backend

# vdvae_extract_features.py 의 H 와 동일 (restore_ema_path 만 절대경로로 바꾼다)
VDVAE_H = {
    'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500, 'save_dir': './saved_models/test',
    'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64',
    'restore_path': 'imagenet64-iter-1600000-model.th',
    'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th',
    'restore_log_path': 'imagenet64-iter-1600000-log.jsonl',
    'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999,
    'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
    'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
    'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64,
    'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0,
    'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0,
    'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0,
    'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000,
    'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8,
    'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0,
    'logdir': './saved_models/test/log',
}
N_VDVAE_LAYERS = 31


class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@contextlib.contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _to_512_tensor(im: Image.Image) -> torch.Tensor:
    """versatilediffusion_reconstruct_images.py 의 regularize_image 와 동일 ([0,1], 3x512x512)."""
    import torchvision.transforms as tvtrans
    im = im.convert("RGB").resize([512, 512], resample=Image.BICUBIC)
    return tvtrans.ToTensor()(im)


class BrainDiffuserBackend(Backend):
    name = "braindiffuser"

    def __init__(self, cfg, need=("vdvae", "vd")):
        self.cfg = cfg
        self.bd_root = cfg.get_path("paths.third_party") / "brain-diffuser"
        if not self.bd_root.exists():
            raise FileNotFoundError(f"{self.bd_root} 가 없다. scripts/00_setup_third_party.sh 를 먼저 실행할 것")
        self.device = torch.device(str(cfg.device))
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 를 쓸 수 없다. GPU 노드에서 실행하거나 backend=mock 으로 점검만 할 것")
            torch.cuda.set_device(self.device)   # VDVAE 와 VD 일부 코드는 기본 cuda 장치를 가정한다
        if not bool(cfg.vd.fp16):
            # 원본 UNetModelVD.forward_dc 가 입력을 직접 half 로 바꾸므로 (openaimodel.py) float32 경로는 동작하지 않는다
            raise ValueError("Versatile Diffusion 은 fp16 으로만 돌릴 수 있다. vd.fp16: true 로 둘 것")
        self.fp16 = True
        self._vae = None
        self._net = None
        self._sampler = None
        self._uim = None
        self._utx = None
        need = set(need)
        if "vdvae" in need:
            self._load_vdvae()
        if "vd" in need or "clip" in need:
            self._load_vd(sampler="vd" in need)

    # ------------------------------------------------------------------ VDVAE
    def _load_vdvae(self) -> None:
        vdvae_dir = self.bd_root / "vdvae"
        ema = vdvae_dir / "model" / "imagenet64-iter-1600000-model-ema.th"
        if not ema.exists():
            raise FileNotFoundError(f"VDVAE 가중치가 없다: {ema}")
        if str(vdvae_dir) not in sys.path:
            sys.path.insert(0, str(vdvae_dir))
        with _cwd(self.bd_root):
            from model_utils import set_up_data, load_vaes  # noqa: E402  (vdvae/model_utils.py)
            H = _DotDict(dict(VDVAE_H))
            H.restore_ema_path = str(ema)
            H.local_rank = self.device.index or 0   # load_vaes 가 ema_vae.cuda(H.local_rank) 로 옮긴다
            H, preprocess_fn = set_up_data(H)
            vae = load_vaes(H)
        vae.eval()
        self._vae, self._preprocess = vae, preprocess_fn
        with torch.no_grad():
            x = torch.zeros((1, 64, 64, 3)).float()
            inp, _ = preprocess_fn(x)
            act = vae.encoder.forward(inp)
            _, stats = vae.decoder.forward(act, get_latents=True)
        self._latent_shapes = [tuple(stats[i]["z"].shape[1:]) for i in range(N_VDVAE_LAYERS)]
        self.vdvae_dim = int(sum(int(np.prod(s)) for s in self._latent_shapes))

    def vdvae_encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torchvision.transforms.functional as TF
        assert self._vae is not None, "VDVAE 가 로드되지 않았다"
        arr = np.stack([np.array(TF.resize(im.convert("RGB"), (64, 64))) for im in images])
        x = torch.tensor(arr).float()
        inp, _ = self._preprocess(x)
        with torch.no_grad():
            act = self._vae.encoder.forward(inp)
            _, stats = self._vae.decoder.forward(act, get_latents=True)
        return np.hstack([stats[i]["z"].cpu().numpy().reshape(len(images), -1)
                          for i in range(N_VDVAE_LAYERS)]).astype(np.float32)

    def vdvae_decode(self, latents: np.ndarray, seed: int = 0) -> List[Image.Image]:
        assert self._vae is not None, "VDVAE 가 로드되지 않았다"
        torch.manual_seed(seed)
        latents = np.atleast_2d(latents)
        n = latents.shape[0]
        layers, off = [], 0
        for shp in self._latent_shapes:
            d = int(np.prod(shp))
            layers.append(torch.tensor(latents[:, off:off + d].reshape((n,) + shp)).float().cuda())
            off += d
        with torch.no_grad():
            px_z = self._vae.decoder.forward_manual_latents(n, layers, t=None)
            samples = self._vae.decoder.out_net.sample(px_z)   # uint8 (n, 64, 64, 3)
        return [Image.fromarray(s).resize((512, 512), resample=Image.BICUBIC) for s in samples]

    # ------------------------------------------------------------------ VD
    def _load_vd(self, sampler: bool = True) -> None:
        vd_dir = self.bd_root / "versatile_diffusion"
        pth = vd_dir / "pretrained" / "vd-four-flow-v1-0-fp16-deprecated.pth"
        if not pth.exists():
            raise FileNotFoundError(f"Versatile Diffusion 가중치가 없다: {pth}")
        if str(vd_dir) not in sys.path:
            sys.path.insert(0, str(vd_dir))
        with _cwd(self.bd_root):   # VD 설정 파일과 kl-f8, optimus 경로가 저장소 루트 기준 상대경로다
            from lib.cfg_helper import model_cfg_bank
            from lib.model_zoo import get_model
            from lib.model_zoo.ddim_vd import DDIMSampler_VD
            cfgm = model_cfg_bank()("vd_noema")
            net = get_model()(cfgm)
            sd = torch.load(str(pth), map_location="cpu")
            net.load_state_dict(sd, strict=False)
        net.clip.to(self.device)
        net.autokl.to(self.device)
        net.autokl.half()
        self._net = net
        if sampler:
            smp = DDIMSampler_VD(net)
            dm = smp.model.model.diffusion_model
            dm.device = str(self.device)
            dm.half()
            dm.to(self.device)
            self._sampler = smp
        self._check_transformers()

    @staticmethod
    def _check_transformers() -> None:
        try:
            import transformers
            major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
            if (major, minor) >= (4, 27):
                print(f"[braindiffuser] 참고: transformers {transformers.__version__}. "
                      "여기서는 CLIP 에 PIL(uint8)을 넘기므로 버전과 무관하게 전처리가 같다.")
        except Exception:
            pass

    def clip_vision(self, images: Sequence[Image.Image]) -> np.ndarray:
        """clipvision_extract_features.py 와 같은 결과. 512px 로 키운 PIL 을 그대로 CLIP 처리기에 넘긴다.

        원본은 [-1,1] 텐서를 넘기고 VD 내부에서 [0,1] numpy 로 되돌리는데, 그 경로는 transformers
        버전에 따라 재스케일이 두 번 적용될 수 있어 PIL 경로를 쓴다. transformers 4.19 의 원본 경로는
        uint8 변환에서 일부 밝기 단계를 1씩 내려 자르므로 값이 아주 조금 다르다. 이 저장소는 특징 추출과
        사슬 모두 이 함수를 쓰므로 안에서는 일관되지만, Brain-Diffuser 가 미리 계산한 특징과 섞어 쓰지는 말 것.
        """
        assert self._net is not None, "VD 가 로드되지 않았다"
        ims = [im.convert("RGB").resize((512, 512), resample=Image.BILINEAR) for im in images]
        with torch.no_grad():
            c = self._net.clip_encode_vision(ims)
        return c.float().cpu().numpy()

    def clip_text(self, captions: Sequence[Sequence[str]]) -> np.ndarray:
        assert self._net is not None, "VD 가 로드되지 않았다"
        out = []
        with torch.no_grad():
            for caps in captions:
                caps = [str(c) for c in caps if str(c) != ""]
                c = self._net.clip_encode_text(caps if caps else [""])
                out.append(c.float().cpu().numpy().mean(0))
        return np.stack(out).astype(np.float32)

    def _uncond(self):
        if self._uim is None:
            dt = torch.float16
            gray = Image.new("RGB", (224, 224), (127, 127, 127))   # 원본: zeros([-1,1]) -> 0.5 회색
            with torch.no_grad():
                self._uim = self._net.clip_encode_vision([gray]).to(self.device).to(dt)
                self._utx = self._net.clip_encode_text([""]).to(self.device).to(dt)
        return self._uim, self._utx

    def generate(self, init_images: Sequence[Image.Image], clip_vision: np.ndarray,
                 clip_text: Optional[np.ndarray], seed: int = 0,
                 strength: Optional[float] = None, mixing: Optional[float] = None) -> List[Image.Image]:
        import torchvision.transforms as tvtrans
        assert self._sampler is not None, "VD 샘플러가 로드되지 않았다"
        vd = self.cfg.vd
        strength = float(vd.strength if strength is None else strength)
        mixing = float(vd.mixing if mixing is None else mixing)
        if clip_text is None:
            mixing = 0.0
        steps = int(vd.ddim_steps)
        dt = torch.float16
        net, sampler = self._net, self._sampler
        uim, utx = self._uncond()
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=float(vd.ddim_eta), verbose=False)
        outs: List[Image.Image] = []
        for i, init in enumerate(init_images):
            torch.manual_seed(seed + i)
            if strength >= 1.0:
                t_enc = int(sampler.ddim_timesteps.shape[0])   # steps 가 1000 의 약수가 아니면 steps 와 다르다
                z_enc = torch.randn((1, 4, 64, 64), device=self.device)
            else:
                zin = (_to_512_tensor(init) * 2 - 1).unsqueeze(0).to(self.device).to(dt)
                with torch.no_grad():
                    init_latent = net.autokl_encode(zin)
                t_enc = int(strength * steps)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(self.device))
            cim = torch.tensor(clip_vision[i:i + 1]).to(self.device).to(dt)
            ctx = utx if clip_text is None else torch.tensor(clip_text[i:i + 1]).to(self.device).to(dt)
            with torch.no_grad():
                z = sampler.decode_dc(
                    x_latent=z_enc,
                    first_conditioning=[uim, cim],
                    second_conditioning=[utx, ctx],
                    t_start=t_enc,
                    unconditional_guidance_scale=float(vd.guidance_scale),
                    xtype="image",
                    first_ctype="vision",
                    second_ctype="prompt",
                    mixed_ratio=(1 - mixing),
                )
                x = net.autokl_decode(z.to(self.device).to(dt))
            x = torch.clamp((x.float() + 1.0) / 2.0, min=0.0, max=1.0)
            outs.append(tvtrans.ToPILImage()(x[0].cpu()))
        return outs
