#!/usr/bin/env bash
# Reproduce the full multi-seed pipeline end-to-end (W14).
#
# Usage:
#   bash scripts/reproduce.sh [config_path]
#
# Defaults to configs/multi_seed_ciai.yaml. Expects:
#   * conda env `tinyml-cancer-forget` already activated
#   * HAM10000 present at dataset.root in the config
#   * 4x GPUs (falls back to single-process when torchrun not available)
set -euo pipefail

CONFIG_PATH="${1:-configs/multi_seed_ciai.yaml}"
NPROC="${NPROC_PER_NODE:-4}"

echo "[reproduce] config: ${CONFIG_PATH}"
echo "[reproduce] GPUs per node: ${NPROC}"

# 1. Prepare processed metadata (idempotent — skips when CSV already exists)
if command -v python >/dev/null; then
  python -m data.download_ham10000 \
      --source-dir "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['dataset']['root'])" "${CONFIG_PATH}")" \
      --output-dir "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['dataset']['root'])" "${CONFIG_PATH}")" \
      || echo "[reproduce] metadata step skipped (already prepared or missing raw files)"
fi

# 2. Main seed sweep (DDP when available)
if command -v torchrun >/dev/null && [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" -m experiments.run_seeds \
      --config "${CONFIG_PATH}" \
      --pillars 0 1 2 3 4 5 6 7 8 9
else
  python -m experiments.run_seeds \
      --config "${CONFIG_PATH}" \
      --pillars 0 1 2 3 4 5 6 7 8 9
fi

# 3. Tier-3 extras — structured sparsity + edge latency
for SEED in $(python -c "import yaml,sys;print(' '.join(str(s) for s in yaml.safe_load(open(sys.argv[1]))['experiment']['seeds']))" "${CONFIG_PATH}"); do
  python -m experiments.e_structured_sparsity --config "${CONFIG_PATH}" --seed "${SEED}" || true
  python -m experiments.e_edge_latency       --config "${CONFIG_PATH}" --seed "${SEED}" || true
done

# 4. Re-aggregate after Tier 3 CSVs land, then tag with clinical regimes
python -m evaluation.aggregate --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"
python -m scripts.clinical_gating --root "$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['logging']['results_dir'])" "${CONFIG_PATH}")"

# 5. Release bundle (optional; no upload performed)
python -m scripts.release --config "${CONFIG_PATH}" --release-dir release || true

echo "[reproduce] done."
