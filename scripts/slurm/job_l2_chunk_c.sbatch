#!/bin/bash
# Bundle M++ L2-full (chunk C) — seeds {6, 7, 8, 9}.
#
# Depends on chunk B (or chunk A, whichever order you chained). All
# four seeds skip pillar 0 and reuse seed 0's dense fine-tune via
# share_finetune_across_seeds. Wall budget: ~14h.
#
#SBATCH --job-name=tinyml_l2_c
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --output=results/slurm_logs/l2_chunk_c_%j.out
#SBATCH --error=results/slurm_logs/l2_chunk_c_%j.err

set -euo pipefail

mkdir -p results/slurm_logs
cd "${SLURM_SUBMIT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || module load miniconda/3 2>/dev/null || true
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-tinyml-cancer-forget}" 2>/dev/null || true

# Defensive: ensure env's bin is on PATH even if conda activate silently
# fails to update it (observed on CSCC compute nodes when the env was
# created by mamba/micromamba but conda is the system anaconda).
ENV_TARGET="${ENV_NAME:-tinyml-cancer-forget}"
if [[ -d "$ENV_TARGET" ]]; then
  ENV_DIR="$ENV_TARGET"
else
  ENV_DIR="$HOME/.local/share/mamba/envs/$ENV_TARGET"
fi
export PATH="$ENV_DIR/bin:$PATH"
echo "[env] ENV_DIR=$ENV_DIR  python=$(which python)"

export NPROC=4
export CONFIG_PATH=configs/isic2019_n10_cluster.yaml
THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC ))
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

echo "[l2_chunk_c] config=${CONFIG_PATH}  seeds=6,7,8,9  GPUs=${NPROC}"
nvidia-smi || true

SEED0_DIR="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['checkpoints_dir'])" "${CONFIG_PATH}")/seed_0"
if [[ ! -d "${SEED0_DIR}" ]]; then
  echo "[l2_chunk_c] FATAL: seed 0 checkpoints directory missing (${SEED0_DIR})."
  exit 2
fi

torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}" \
    --seeds 6 7 8 9

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[l2_chunk_c] done"
