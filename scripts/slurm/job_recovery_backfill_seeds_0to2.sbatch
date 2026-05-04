#!/bin/bash
# Bundle M++ — Recovery backfill for seeds {0, 1, 2}.
#
# Background:
#   P0.1 diagnostic (scripts/diagnose_recovery.py) found that the
#   personal-compute recovery sweep at lr=1e-5 was a no-op — weights
#   never updated between e5/e10/e20 checkpoints. The new n10 cluster
#   config raises recovery LR to 1e-4 and sweeps {1e-4, 5e-4, 1e-3} so
#   seeds 3..9 will train properly.
#
#   For n=10 aggregation consistency, this job re-runs pillar 5
#   (recovery_finetune) for the rsync'd seeds 0, 1, 2 against the n10
#   config so all 10 seeds use the same recovery hyper-parameters.
#
#   The pre-existing pillars 0..4 outputs (dense ckpt, masks, pruning
#   matrix, etc.) on seeds 0..2 are reused as-is — only pillar 5 is
#   re-run. Broken recovery_*.pth and recovery_finetune*.csv files
#   under those seeds were already deleted by the operator.
#
# Submit (independent of S0; can run in parallel as cluster allows):
#   sbatch scripts/slurm/job_recovery_backfill_seeds_0to2.sbatch
#
#SBATCH --job-name=tinyml_rec_bf
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm_logs/recovery_backfill_%j.out
#SBATCH --error=results/slurm_logs/recovery_backfill_%j.err

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

echo "[recovery_backfill] config=${CONFIG_PATH}  seeds=0,1,2  pillar=5  GPUs=${NPROC}"
nvidia-smi || true

# Preflight — make sure pillars 0..4 outputs are present (linked from
# results/checkpoints_personal/) and the broken recovery files were
# cleaned up.
for SEED in 0 1 2; do
  CKPT_DIR="results/checkpoints_n10/seed_${SEED}"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[recovery_backfill] FATAL: ${CKPT_DIR} missing. Did you symlink seeds 0..2 from checkpoints_personal?"
    exit 2
  fi
  if [[ ! -f "${CKPT_DIR}/deit_small_ham10000.pth" ]]; then
    echo "[recovery_backfill] FATAL: ${CKPT_DIR}/deit_small_ham10000.pth missing — pillar 0 outputs not present."
    exit 3
  fi
  STALE_RECOVERY=$(find "${CKPT_DIR}" -maxdepth 1 -name 'recovery_*.pth' 2>/dev/null | wc -l)
  if [[ "${STALE_RECOVERY}" -gt 0 ]]; then
    echo "[recovery_backfill] WARNING: ${STALE_RECOVERY} recovery_*.pth files still present in ${CKPT_DIR}."
    echo "                    Pillar 5 will overwrite, but you may want to confirm they were the broken ones."
  fi
done

# Run pillar 5 (recovery sweep) for seeds 0, 1, 2 only.
torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
    --config "${CONFIG_PATH}" \
    --pillars 5 \
    --seeds 0 1 2

# Re-aggregate so the new recovery numbers feed the n=10 tables.
python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[recovery_backfill] done"
