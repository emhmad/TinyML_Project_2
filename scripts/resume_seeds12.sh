#!/usr/bin/env bash
# Run seeds 1 and 2 only (seed 0 is already complete) and then
# aggregate + clinical-gate. Uses pipefail so any python failure
# halts the script instead of being masked by tee.
set -eo pipefail

source .venv/bin/activate

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

TS=$(date +%Y%m%d_%H%M%S)

python -m experiments.run_seeds \
    --config configs/local_personal_8h.yaml \
    --seeds 1 2 \
    2>&1 | tee -a "results/logs_personal/resume_seeds12_${TS}.log"

python -m evaluation.aggregate --root results/logs_personal || true
python -m scripts.clinical_gating --root results/logs_personal || true

echo "ALL DONE"
