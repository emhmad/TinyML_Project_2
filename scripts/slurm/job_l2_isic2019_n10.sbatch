#!/bin/bash
# Bundle M++ L2-full — ISIC 2019 retrained sweep at n=10.
#
# DEPRECATED: this is the single 48h mega-job version. cscc-gpu-p has
# PreemptMode=REQUEUE, so a 48h job that gets preempted at hour 47
# wastes ~47 GPU-hours and starts over. Prefer the chained chunks:
#     job_l2_chunk_a.sbatch  (seeds 0..2, ~16h)
#     job_l2_chunk_b.sbatch  (seeds 3..5, ~12h, afterok chunk A)
#     job_l2_chunk_c.sbatch  (seeds 6..9, ~14h, afterok chunk B)
# The orchestrator scripts/run_bundle_m_plus.sh uses the chunked path.
#
# Prerequisite: data/isic2019/processed_metadata.csv built via
# `python -m scripts.build_isic2019_metadata --source-dir data/isic2019 \
#                                            --output-dir data/isic2019`
#
# Submit (escape hatch only):
#   sbatch scripts/slurm/job_l2_isic2019_n10.sbatch
#
#SBATCH --job-name=tinyml_l2_isic2019
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=results/slurm_logs/l2_isic2019_%j.out
#SBATCH --error=results/slurm_logs/l2_isic2019_%j.err

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

echo "[l2_isic2019] config=${CONFIG_PATH}  GPUs=${NPROC}"
nvidia-smi || true

# Sanity-check: bail loudly if metadata wasn't built.
META_CSV="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['dataset']['metadata_csv'])" "${CONFIG_PATH}")"
if [[ ! -f "${META_CSV}" ]]; then
  echo "[l2_isic2019] FATAL: ISIC 2019 metadata not found at ${META_CSV}."
  echo "Run: python -m scripts.build_isic2019_metadata --source-dir data/isic2019 --output-dir data/isic2019"
  exit 2
fi

torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}"

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[l2_isic2019] done"
