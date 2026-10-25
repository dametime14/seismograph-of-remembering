#!/usr/bin/env bash
# NSD (Allen et al. 2022) 에서 한 피험자의 자료를 받는다. 기본은 subj01.
#
#   NSD_ROOT=/storage/me/nsd bash scripts/01_download_nsd.sh 1
#
# NSD 는 공개 S3 버킷(us-east-2)에서 계정 없이 받을 수 있지만, 사용 전에 NSD Data Access Agreement 를
# 작성하고 이용 조건을 따라야 한다: https://naturalscenesdataset.org
#
# 받는 것 (subj01 기준 대략)
#   nsd_expdesign.mat, nsd_stim_info_merged.pkl     실험 설계
#   nsd_stimuli.hdf5                                 73,000장 자극 (약 37GB)
#   betas_session01..37.nii.gz                       func1pt8mm, betas_fithrf_GLMdenoise_RR (세션당 750시행)
#   nsdgeneral.nii.gz                                시각 피질 ROI
#   COCO_73k_annots_curated.npy                      COCO 캡션 (CLIP-Text 디코더용, HuggingFace pscotti/naturalscenesdataset)
set -euo pipefail
cd "$(dirname "$0")/.."

SUB="${1:-1}"
NSD_ROOT="${NSD_ROOT:-./data/nsd}"
N_SESS="${N_SESS:-37}"
S3=s3://natural-scenes-dataset
SUBJ=$(printf "subj%02d" "$SUB")
AWS="aws s3 cp --no-sign-request --only-show-errors"

command -v aws >/dev/null || { echo "aws CLI 가 필요하다 (pip install awscli 또는 conda install -c conda-forge awscli)"; exit 1; }

mkdir -p "$NSD_ROOT/nsddata/experiments/nsd" "$NSD_ROOT/nsddata_stimuli/stimuli/nsd" \
         "$NSD_ROOT/nsddata/ppdata/$SUBJ/func1pt8mm/roi" \
         "$NSD_ROOT/nsddata_betas/ppdata/$SUBJ/func1pt8mm/betas_fithrf_GLMdenoise_RR"

get () { [ -s "$2" ] && echo "[nsd] exists: $(basename "$2")" || { echo "[nsd] $1"; $AWS "$S3/$1" "$2"; }; }

get nsddata/experiments/nsd/nsd_expdesign.mat        "$NSD_ROOT/nsddata/experiments/nsd/nsd_expdesign.mat"
get nsddata/experiments/nsd/nsd_stim_info_merged.pkl "$NSD_ROOT/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
get nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5     "$NSD_ROOT/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
get "nsddata/ppdata/$SUBJ/func1pt8mm/roi/nsdgeneral.nii.gz" "$NSD_ROOT/nsddata/ppdata/$SUBJ/func1pt8mm/roi/nsdgeneral.nii.gz"

for s in $(seq 1 "$N_SESS"); do
  f=$(printf "betas_session%02d.nii.gz" "$s")
  get "nsddata_betas/ppdata/$SUBJ/func1pt8mm/betas_fithrf_GLMdenoise_RR/$f" \
      "$NSD_ROOT/nsddata_betas/ppdata/$SUBJ/func1pt8mm/betas_fithrf_GLMdenoise_RR/$f"
done

ANN="$NSD_ROOT/COCO_73k_annots_curated.npy"
if [ ! -s "$ANN" ]; then
  echo "[nsd] COCO_73k_annots_curated.npy"
  URL=https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/COCO_73k_annots_curated.npy
  if command -v wget >/dev/null; then wget -q --show-progress -c -O "$ANN" "$URL"; else curl -L -C - -o "$ANN" "$URL"; fi
fi
echo "[nsd] done: $NSD_ROOT"
