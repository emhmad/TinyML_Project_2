#!/bin/bash
# Job A — runs seeds 0 and 2 (core pillars on both + seed-zero-only
# pillars 7/9/11 on seed 0). ~6–8 h wall.
#
# Submit:
#   sbatch scripts/slurm/job_a_seeds_0_2.sbatch
#
#SBATCH --job-name=tinyml_job_a
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm_logs/job_a_%j.out
#SBATCH --error=results/slurm_logs/job_a_%j.err

set -euo pipefail

mkdir -p results/slurm_logs

cd "${SLURM_SUBMIT_DIR}"

# --- conda activation ---
if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || module load miniconda/3 2>/dev/null || true
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-tinyml-cancer-forget}"

# --- runtime knobs ---
export NPROC=4
export CONFIG_PATH=configs/multi_seed_ciai_fast.yaml

# Multi-threading: pin OMP / MKL / numexpr so each DDP process has a
# sane slice of the 32 allocated CPUs. 32 CPUs / 4 GPU processes = 8
# threads per process for DataLoader + BLAS.
THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC ))
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export TOKENIZERS_PARALLELISM=false
# NCCL tuning — lets DDP use the fastest available inter-GPU transport
# on A100 nodes. Harmless on other hardware.
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# Keep tqdm on a single line per epoch so SLURM .out files stay readable.
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

echo "[job_a] seeds=0 2  GPUs=${NPROC}  threads/proc=${THREADS_PER_PROC}"
nvidia-smi || true

# --- run ---
SEEDS="0 2" bash scripts/reproduce_fast.sh
