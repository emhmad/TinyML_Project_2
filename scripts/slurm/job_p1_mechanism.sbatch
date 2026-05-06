#!/bin/bash
# Bundle M++ P1.1 + P1.3 — mechanism probes for the DCR inversion.
#
# Runs the per-class gradient flow (P1.1) and per-attention-head
# class-conditional activation (P1.3) experiments across all n=10 seeds.
# Both are inference + small-backward operations — single GPU is plenty.
#
# Depends on S0 having produced the dense fine-tuned checkpoints and
# the saved pruning masks under results/checkpoints_n10/seed_*/masks/.
#
# Time budget: ~4–6 h on 1 A100 across n=10 seeds.
#
# Submit:
#   sbatch --dependency=afterok:<S0_JOB_ID> scripts/slurm/job_p1_mechanism.sbatch
#
#SBATCH --job-name=tinyml_p1_mech
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=results/slurm_logs/p1_mechanism_%j.out
#SBATCH --error=results/slurm_logs/p1_mechanism_%j.err

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

export CONFIG_PATH=configs/multi_seed_n10_cluster.yaml
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "[p1_mechanism] config=${CONFIG_PATH}"
nvidia-smi || true

for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "[p1_mechanism] seed=${SEED}  P1.1 (per-class gradient flow)"
  python -m experiments.e_perclass_gradient_flow --config "${CONFIG_PATH}" --seed "${SEED}"

  echo "[p1_mechanism] seed=${SEED}  P1.3 (per-head class-conditional activation)"
  python -m experiments.e_perhead_activation --config "${CONFIG_PATH}" --seed "${SEED}"
done

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[p1_mechanism] done"
