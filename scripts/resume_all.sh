#!/usr/bin/env bash
# Resume the remaining pieces of the personal 8h sweep:
#   1. Seed 0 pillars 7, 9, 11 (MobileNetV2 + activation stats / Paxton / X-Pruner + edge latency)
#   2. Seeds 1 and 2, full pillar list (0 auto-skipped by share_finetune_across_seeds)
#   3. Aggregation + clinical gating
#
# set -o pipefail is essential: without it `python ... | tee log` always
# returns tee's zero exit code, and a python failure wouldn't halt the
# script. Every step below is expected to either exit 0 or surface a
# real error.
set -eo pipefail

source .venv/bin/activate

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# torch.onnx.export in torch>=2.6 pulls in onnxscript. Add it if missing
# so pillar 11 (edge_latency) doesn't crash.
python -c "import onnxscript" 2>/dev/null || pip install --quiet onnxscript

rm -f results/logs_personal/seed_0/mobilenet_results.csv
rm -f results/logs_personal/seed_0/activation_stats.csv
rm -f results/logs_personal/seed_0/activation_stats_correlation.csv
rm -f results/logs_personal/seed_0/activation_stats_layerwise_damage.csv
rm -f results/logs_personal/seed_0/baselines_paxton_xpruner.csv
rm -f results/logs_personal/seed_0/edge_latency.csv

TS=$(date +%Y%m%d_%H%M%S)

python -m experiments.run_seeds \
    --config configs/local_personal_8h.yaml \
    --seeds 0 --pillars 7 9 11 \
    2>&1 | tee -a "results/logs_personal/resume_seed0_${TS}.log"

python -m experiments.run_seeds \
    --config configs/local_personal_8h.yaml \
    --seeds 1 2 \
    2>&1 | tee -a "results/logs_personal/resume_seeds12_${TS}.log"

python -m evaluation.aggregate --root results/logs_personal || true
python -m scripts.clinical_gating --root results/logs_personal || true

echo "ALL DONE"
