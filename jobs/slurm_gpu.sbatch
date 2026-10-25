#!/bin/bash
# Slurm 이 있는 서버용 틀. 계정, 파티션, 제약 조건은 서버에 맞게 고칠 것.
# NERSC Perlmutter 예:  -A <ACCOUNT> -C gpu -q regular
#
#   sbatch jobs/slurm_gpu.sbatch scripts/03_extract_features.py
#   sbatch --array=0-3 jobs/slurm_gpu.sbatch scripts/08_run_chains.py --num-shards 4
#
#SBATCH -J seismograph
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
##SBATCH -A <ACCOUNT>
##SBATCH -C gpu
##SBATCH -q regular

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
CONFIG="${CONFIG:-configs/local.yaml}"
# 환경 활성화 (예시). 자신의 환경 이름으로 바꿀 것
# module load conda && conda activate seismograph

SCRIPT="$1"; shift
EXTRA=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  EXTRA+=(--shard "$SLURM_ARRAY_TASK_ID")
fi
python "$SCRIPT" --config "$CONFIG" "$@" "${EXTRA[@]}"
