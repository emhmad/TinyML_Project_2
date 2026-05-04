#!/bin/bash
# Bundle M++ L3-full (chunk B) — ResNet-50 baseline at n=10.
#
# Pillar 8 only (e_resnet50_baseline). ResNet-50 is the larger backbone
# and was previously dropped for compute reasons; this chunk brings it
# back at n=10. Wall budget: ~16h on 4x A100.
#
#SBATCH --job-name=tinyml_l3_b_r50
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/l3_chunk_b_%j.out
#SBATCH --error=results/slurm_logs/l3_chunk_b_%j.err

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

echo "[l3_chunk_b] config=${CONFIG_PATH}  pillar=8 (ResNet-50)  GPUs=${NPROC}"
nvidia-smi || true

torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}" \
    --pillars 8

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[l3_chunk_b] done"
