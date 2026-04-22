#!/bin/bash
# Job B — runs seed 1 only. ~3 h wall. Submit concurrently with Job A
# once IT lifts the 1-job-at-a-time cap (see scripts/it_request_concurrent_jobs.txt).
#
# Submit:
#   sbatch scripts/slurm/job_b_seed_1.sbatch
#
#SBATCH --job-name=tinyml_job_b
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm_logs/job_b_%j.out
#SBATCH --error=results/slurm_logs/job_b_%j.err

set -euo pipefail

mkdir -p results/slurm_logs

cd "${SLURM_SUBMIT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || module load miniconda/3 2>/dev/null || true
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-tinyml-cancer-forget}"

export NPROC=4
export CONFIG_PATH=configs/multi_seed_ciai_fast.yaml

THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC ))
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

echo "[job_b] seeds=1  GPUs=${NPROC}  threads/proc=${THREADS_PER_PROC}"
nvidia-smi || true

SEEDS="1" bash scripts/reproduce_fast.sh
