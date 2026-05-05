#!/bin/bash
# Bundle M++ L4-full — quantitative attention overlap (W12).
#
# Reuses the existing experiments.e_attention_viz module — it already
# computes IoU + pointing-game + mass-in-mask via
# evaluation.attention_overlap. We just need ISIC 2018 segmentation
# masks staged at dataset.segmentation_mask_dir, which the per-job
# preflight verifies below.
#
# Depends on S0 having produced the n=10 pruned checkpoints + masks for
# magnitude / wanda / taylor at sparsity 0.5.
#
# Time budget: ~4–8 h on 4x A100 (mostly inference + figure generation).
#
# Submit:
#   sbatch --dependency=afterok:<S0_JOB_ID> scripts/slurm/job_l4_attention_iou.sbatch
#
#SBATCH --job-name=tinyml_l4_iou
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm_logs/l4_iou_%j.out
#SBATCH --error=results/slurm_logs/l4_iou_%j.err

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

# Preflight: verify the segmentation-mask directory has masks staged.
MASK_DIR="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('dataset',{}).get('segmentation_mask_dir',''))" "${CONFIG_PATH}")"
if [[ -z "${MASK_DIR}" || ! -d "${MASK_DIR}" ]]; then
  echo "[l4_iou] FATAL: segmentation_mask_dir not configured or missing at '${MASK_DIR}'."
  echo "Run: python -m scripts.build_isic2018_segmasks \\"
  echo "       --source-dir <ISIC_2018_Task1_GroundTruth_dir> \\"
  echo "       --metadata-csv data/ham10000/processed_metadata.csv \\"
  echo "       --output-dir data/ham10000/segmentation_masks"
  exit 2
fi
N_MASKS=$(find "${MASK_DIR}" -maxdepth 1 -name '*.png' | wc -l)
echo "[l4_iou] mask dir: ${MASK_DIR}  (${N_MASKS} masks staged)"
if [[ "${N_MASKS}" -lt 100 ]]; then
  echo "[l4_iou] FATAL: fewer than 100 masks staged. Re-run scripts.build_isic2018_segmasks."
  exit 3
fi

# Run the W12 attention-overlap experiment for each seed.
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "[l4_iou] seed=${SEED}"
  python -m experiments.e_attention_viz --config "${CONFIG_PATH}" --seed "${SEED}"
done

python -m evaluation.aggregate \
    --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

echo "[l4_iou] done"
