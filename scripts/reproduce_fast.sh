#!/usr/bin/env bash
# Fast 3-seed reproduction driver for CSCC/CIAI.
#
# Produces both a short stdout trail (captured by SLURM .out) and a
# timestamped, full-fidelity log file under the results logs directory
# so post-hoc debugging doesn't depend on SLURM retention.
#
# Two modes:
#   (a) Single 4-GPU job, sequential seeds (~9 h wall)
#       bash scripts/reproduce_fast.sh
#   (b) Two concurrent 4-GPU jobs (~6 h wall) — see scripts/slurm/*.
#
# Env vars:
#   CONFIG_PATH  - yaml config (default: configs/multi_seed_ciai_fast.yaml)
#   SEEDS        - optional space-separated seeds to run (subset of config)
#   PILLARS      - optional space-separated pillar ids (override config)
#   NPROC        - GPUs per node (default 4)
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/multi_seed_ciai_fast.yaml}"
NPROC="${NPROC:-4}"

RESULTS_DIR="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RESULTS_DIR}/run_${TIMESTAMP}_${SLURM_JOB_ID:-local}.log"

# Everything below gets both to stdout (which SLURM will capture into
# the .out file) AND to a persistent log file under RESULTS_DIR.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[reproduce_fast] config=${CONFIG_PATH}"
echo "[reproduce_fast] GPUs=${NPROC}  seeds=${SEEDS:-<from config>}  pillars=${PILLARS:-<from config>}"
echo "[reproduce_fast] run log: ${RUN_LOG}"
echo "[reproduce_fast] host=$(hostname)  date=$(date)"

SEED_ARGS=()
if [[ -n "${SEEDS:-}" ]]; then
  SEED_ARGS+=(--seeds ${SEEDS})
fi
PILLAR_ARGS=()
if [[ -n "${PILLARS:-}" ]]; then
  PILLAR_ARGS+=(--pillars ${PILLARS})
fi

# HAM10000 metadata preparation (idempotent — skipped if processed CSV exists)
DATASET_ROOT="$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['dataset']['root'])" "${CONFIG_PATH}")"
if [[ ! -f "${DATASET_ROOT}/processed_metadata.csv" ]]; then
  echo "[reproduce_fast] building processed_metadata.csv ..."
  python -m data.download_ham10000 --source-dir "${DATASET_ROOT}" --output-dir "${DATASET_ROOT}" \
    || echo "[reproduce_fast] metadata step failed — verify raw data is present."
fi

# Main sweep
if command -v torchrun >/dev/null && [[ "${NPROC}" -gt 1 ]]; then
  echo "[reproduce_fast] launching via torchrun --nproc_per_node=${NPROC}"
  torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
      --config "${CONFIG_PATH}" "${SEED_ARGS[@]}" "${PILLAR_ARGS[@]}"
else
  echo "[reproduce_fast] launching single-process python"
  python -m experiments.run_seeds --config "${CONFIG_PATH}" "${SEED_ARGS[@]}" "${PILLAR_ARGS[@]}"
fi

echo "[reproduce_fast] aggregating across seeds under ${RESULTS_DIR}"
python -m evaluation.aggregate --root "${RESULTS_DIR}"
python -m scripts.clinical_gating --root "${RESULTS_DIR}" || true

echo "[reproduce_fast] done. logs: ${RUN_LOG}"
