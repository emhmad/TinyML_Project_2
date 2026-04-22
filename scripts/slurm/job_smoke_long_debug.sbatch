#!/bin/bash
# Smoke test on partition=long with gpu-debug-qos (3h max).
# Runs only pillars 0 and 1 (fine-tune + pruning matrix) on seed 0 so
# we verify env + DDP + data loading + evaluation end-to-end within
# the debug walltime. Does NOT produce the full paper results.
#
#SBATCH --job-name=tinyml_smoke
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --output=results/slurm_logs/smoke_%j.out
#SBATCH --error=results/slurm_logs/smoke_%j.err

set -euo pipefail
mkdir -p results/slurm_logs
cd "${SLURM_SUBMIT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || module load miniconda/3 2>/dev/null || true
fi

# If you're using the venv path, activate that instead.
if [[ -d "${HOME}/venvs/tinyml" ]]; then
  source "${HOME}/venvs/tinyml/bin/activate"
else
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME:-tinyml-cancer-forget}"
fi

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

echo "[smoke] seed=0 pillars=0,1 (fine-tune + pruning matrix only)"
nvidia-smi || true

SEEDS="0" PILLARS="0 1" bash scripts/reproduce_fast.sh
