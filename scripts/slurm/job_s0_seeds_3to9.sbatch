#!/bin/bash
# Bundle M++ S0 — extends pruning matrix to seeds 3..9.
#
# Seeds 0..2 are preserved from the existing personal/cluster runs;
# this job adds the additional seven seeds so every downstream M++
# analysis (P1.1, P1.3, L4 IoU) has n=10. Per-seed marginal cost is
# ~1 h on 4x A100 with share_finetune_across_seeds=true.
#
# Submit:
#   sbatch scripts/slurm/job_s0_seeds_3to9.sbatch
#
#SBATCH --job-name=tinyml_s0_n10
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=results/slurm_logs/s0_seeds_3to9_%j.out
#SBATCH --error=results/slurm_logs/s0_seeds_3to9_%j.err

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

# --- runtime knobs ---
export NPROC=4
export CONFIG_PATH=configs/multi_seed_n10_cluster.yaml
THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC ))
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

echo "[s0_seeds_3to9] config=${CONFIG_PATH}  GPUs=${NPROC}  threads/proc=${THREADS_PER_PROC}"
nvidia-smi || true

# Run only the new seeds. Pillars come from config (experiment.pillars).
torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}" \
    --seeds 3 4 5 6 7 8 9

# Re-aggregate after the new seeds land. The aggregator picks up the
# existing seeds 0..2 (if their results are merged into the same root —
# see scripts/run_bundle_m_plus.sh for the merge step).
python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[s0_seeds_3to9] done"
