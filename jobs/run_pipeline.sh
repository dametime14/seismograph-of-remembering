#!/usr/bin/env bash
# 랩서버(작업 스케줄러 없는 GPU 서버)에서 1부 전체를 순서대로 돌린다.
#
#   CUDA_VISIBLE_DEVICES=0 CONFIG=configs/local.yaml bash jobs/run_pipeline.sh
#   FROM=4 bash jobs/run_pipeline.sh       # 4단계(디코더)부터 다시
#
# 단계별 자원 (subj01 기준 추정. 처음 돌릴 때 시간과 메모리를 재서 README 에 적어 둘 것)
#   02 준비        CPU, RAM 16GB 이상, 디스크 I/O 위주
#   03 특징        GPU (VDVAE + CLIP, 학습 8,859장 + 테스트 982장), 특징 파일 약 14GB
#   04 디코더      GPU 또는 CPU, RAM 32GB 이상, 가중치 약 25GB 저장 (대조 사슬용 사상 포함)
#   05 인코더      GPU 또는 CPU, RAM 32GB 이상, 가중치 약 6.5GB 저장
#   06 점검        GPU
#   07 보정        CPU 또는 GPU
#   08 사슬        GPU. 사슬 한 단계 = VDVAE 복원 1번 + VD 확산 37스텝. 전체 단계 수 = 그림 x 조건 x 반복 x n_iters
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/local.yaml}"
FROM="${FROM:-2}"
run () { echo; echo "===== $* ====="; python "$@" --config "$CONFIG"; }

[ "$FROM" -le 2 ] && run scripts/02_prepare_nsd.py
[ "$FROM" -le 3 ] && run scripts/03_extract_features.py
[ "$FROM" -le 4 ] && run scripts/04_train_decoders.py
[ "$FROM" -le 5 ] && run scripts/05_train_encoder.py
[ "$FROM" -le 6 ] && run scripts/06_sanity_reconstruct.py
[ "$FROM" -le 7 ] && run scripts/07_calibrate.py
[ "$FROM" -le 8 ] && run scripts/08_run_chains.py
echo "[pipeline] done"
