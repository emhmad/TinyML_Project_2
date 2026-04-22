#!/usr/bin/env bash
# Personal-compute driver — single-machine, 8-hour budget.
#
# What it does:
#   1. Activates the local venv/conda env (edit ACTIVATE_CMD below).
#   2. Verifies the dataset + metadata CSV exist; prints instructions
#      if not.
#   3. Runs the 3-seed sweep with experiment.share_finetune_across_seeds
#      so only seed 0 trains, seeds 1 & 2 reuse the checkpoint.
#   4. Aggregates + clinical-gates + tags a summary banner at the end.
#
# Typical walltime (M3 Pro/Max MacBook): ~6–7 h.
# Typical walltime (RTX 4090 desktop):   ~3–4 h.
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/local_personal_8h.yaml}"
ACTIVATE_CMD="${ACTIVATE_CMD:-source ~/venvs/tinyml/bin/activate}"

cd "$(dirname "$0")/.."

# Best-effort env activation. Edit ACTIVATE_CMD above if your env lives elsewhere.
# shellcheck disable=SC1090
eval "${ACTIVATE_CMD}" || echo "[warn] could not activate env with '${ACTIVATE_CMD}' — continuing with current shell"

# Sanity: dataset + metadata present?
DATASET_ROOT="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['dataset']['root'])" "${CONFIG_PATH}")"
META_CSV="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['dataset']['metadata_csv'])" "${CONFIG_PATH}")"

if [[ ! -f "${META_CSV}" ]]; then
  echo "[info] ${META_CSV} not found — building it now."
  python -m data.download_ham10000 --source-dir "${DATASET_ROOT}" --output-dir "${DATASET_ROOT}"
fi

# Enable MPS fallback for any ops not yet supported on Apple Silicon.
# This auto-falls-back to CPU for the missing op, avoiding hard crashes.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

# CPU thread budget — tune to your machine (M3 Pro = 11 cores, M3 Max = 14).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"

# Timestamped log alongside results.
RESULTS_DIR="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[personal] config=${CONFIG_PATH}  log=${LOG}"
python -c "import torch, platform; print('[personal] python', platform.python_version(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"

python -m experiments.run_seeds --config "${CONFIG_PATH}"

echo "[personal] aggregating..."
python -m evaluation.aggregate --root "${RESULTS_DIR}"
python -m scripts.clinical_gating --root "${RESULTS_DIR}" || true

echo "[personal] done. Look under ${RESULTS_DIR}/aggregated/ for mean±std tables and paired tests."
