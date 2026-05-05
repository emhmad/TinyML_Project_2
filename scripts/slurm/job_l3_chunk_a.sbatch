#!/bin/bash
# Bundle M++ L3-full (chunk A) — MobileNetV2 baseline at n=10.
#
# Pillar 7 only (e_mobilenet_baseline). Independent of chunk B.
# Wall budget: ~16h on 4x A100. Each seed runs its own dense CNN
# fine-tune (share_finetune_across_seeds=false in the L3 config).
#
# Submit (chained):
#   L3A=$(sbatch --parsable scripts/slurm/job_l3_chunk_a.sbatch)
#   L3B=$(sbatch --parsable --dependency=afterok:$L3A scripts/slurm/job_l3_chunk_b.sbatch)
#
#SBATCH --job-name=tinyml_l3_a_mn2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/l3_chunk_a_%j.out
#SBATCH --error=results/slurm_logs/l3_chunk_a_%j.err

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
export CONFIG_PATH=configs/l3_cnn_n10_cluster.yaml
THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC ))
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

echo "[l3_chunk_a] config=${CONFIG_PATH}  pillar=7 (MobileNetV2)  GPUs=${NPROC}"
nvidia-smi || true

torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}" \
    --pillars 7

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[l3_chunk_a] done"
