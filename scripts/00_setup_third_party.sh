#!/usr/bin/env bash
# Brain-Diffuser 원본 코드와 사전학습 가중치를 받는다.
#
#   bash scripts/00_setup_third_party.sh
#   WEIGHTS_DIR=/storage/me/weights bash scripts/00_setup_third_party.sh   # 가중치를 큰 디스크에
#
# 받는 것 (합계 약 8.5GB)
#   third_party/brain-diffuser            ozcelikfu/brain-diffuser (MIT), 고정 커밋
#   VDVAE ImageNet64 EMA 가중치             openaipublic (Child 2021)
#   Versatile Diffusion v1.0 fp16 (deprecated 판), kl-f8, optimus-vae   HuggingFace shi-labs/versatile-diffusion
#   CLIP ViT-L/14                          HuggingFace openai/clip-vit-large-patch14 (transformers 캐시)
#
# 계산 노드에 인터넷이 없으면 로그인 노드에서 이 스크립트를 먼저 돌려 둘 것.
set -euo pipefail
cd "$(dirname "$0")/.."

BD_COMMIT=31d52c01b7f204b9887024eb670065a0197f5bcc
WEIGHTS_DIR="${WEIGHTS_DIR:-./data/weights}"
mkdir -p third_party "$WEIGHTS_DIR/vdvae" "$WEIGHTS_DIR/versatile_diffusion"
WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"

# 1) Brain-Diffuser 코드
if [ ! -d third_party/brain-diffuser/.git ]; then
  git clone https://github.com/ozcelikfu/brain-diffuser.git third_party/brain-diffuser
fi
git -C third_party/brain-diffuser fetch -q origin || true
git -C third_party/brain-diffuser checkout -q "$BD_COMMIT"
echo "[setup] brain-diffuser @ $BD_COMMIT"

fetch () {  # url dest
  if [ -s "$2" ]; then echo "[setup] exists: $2"; return; fi
  echo "[setup] downloading $(basename "$2")"
  if command -v wget >/dev/null; then wget -q --show-progress -c -O "$2" "$1"; else curl -L -C - -o "$2" "$1"; fi
}

# 2) VDVAE: 원본 README 는 4개 파일을 받지만 실제로 쓰는 것은 EMA 가중치뿐이다
VDVAE_URL=https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2
for f in imagenet64-iter-1600000-model-ema.th imagenet64-iter-1600000-log.jsonl; do
  fetch "$VDVAE_URL/$f" "$WEIGHTS_DIR/vdvae/$f"
done

# 3) Versatile Diffusion
HF_VD=https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth
for f in vd-four-flow-v1-0-fp16-deprecated.pth kl-f8.pth optimus-vae.pth; do
  fetch "$HF_VD/$f" "$WEIGHTS_DIR/versatile_diffusion/$f"
done

# 4) brain-diffuser 가 기대하는 상대경로에 심볼릭 링크
mkdir -p third_party/brain-diffuser/vdvae/model third_party/brain-diffuser/versatile_diffusion/pretrained
for f in imagenet64-iter-1600000-model-ema.th imagenet64-iter-1600000-log.jsonl; do
  ln -sf "$WEIGHTS_DIR/vdvae/$f" "third_party/brain-diffuser/vdvae/model/$f"
done
for f in vd-four-flow-v1-0-fp16-deprecated.pth kl-f8.pth optimus-vae.pth; do
  ln -sf "$WEIGHTS_DIR/versatile_diffusion/$f" "third_party/brain-diffuser/versatile_diffusion/pretrained/$f"
done

# 5) CLIP (VD 가 transformers 로 불러온다). 실패해도 첫 실행 때 다시 시도된다.
python - <<'PY' || echo "[setup] CLIP 캐시를 만들지 못했다. 인터넷이 되는 노드에서 다시 실행할 것"
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
name = "openai/clip-vit-large-patch14"
CLIPTokenizer.from_pretrained(name); CLIPProcessor.from_pretrained(name); CLIPModel.from_pretrained(name)
print("[setup] CLIP cached")
PY

echo "[setup] done. weights in $WEIGHTS_DIR"
