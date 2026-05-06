#!/bin/bash
# Bundle M++ L3-full — MobileNetV2 + ResNet-50 baselines at n=10.
#
# DEPRECATED: this is the single 36h mega-job version. cscc-gpu-p has
# PreemptMode=REQUEUE — long jobs are at risk of restart-from-scratch.
# Prefer the chained chunks:
#     job_l3_chunk_a.sbatch  (MobileNetV2, ~16h)
#     job_l3_chunk_b.sbatch  (ResNet-50,   ~16h, afterok chunk A)
# The orchestrator scripts/run_bundle_m_plus.sh uses the chunked path.
#
# Submit (escape hatch only):
#   sbatch scripts/slurm/job_l3_cnn_n10.sbatch
#
#SBATCH --job-name=tinyml_l3_cnn
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=results/slurm_logs/l3_cnn_%j.out
#SBATCH --error=results/slurm_logs/l3_cnn_%j.err

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

echo "[l3_cnn] config=${CONFIG_PATH}  GPUs=${NPROC}"
nvidia-smi || true

torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}"

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[l3_cnn] done"
