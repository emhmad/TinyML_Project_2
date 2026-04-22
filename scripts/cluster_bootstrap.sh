#!/usr/bin/env bash
# One-shot environment bootstrap on CIAI / CSCC.
#
# Creates (or updates) a conda env named `tinyml-cancer-forget`, installs
# every pip dependency, then verifies CUDA visibility and key imports.
#
# Idempotent: re-running only picks up new dependencies.
#
# IMPORTANT: run on a login node. This script does NOT launch GPU jobs —
# training is submitted via scripts/slurm/*.sbatch.
set -euo pipefail

ENV_NAME="${ENV_NAME:-tinyml-cancer-forget}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "${REPO_ROOT}"

echo "[bootstrap] repo=${REPO_ROOT}"
echo "[bootstrap] env=${ENV_NAME}"
echo "[bootstrap] host=$(hostname)"

# MBZUAI HPC provides miniconda via `module`, but falls back to direct
# binary if the module system isn't configured.
if command -v module >/dev/null 2>&1; then
  module load anaconda3 2>/dev/null || module load miniconda/3 2>/dev/null || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[bootstrap] conda not found on PATH. Install miniconda in \$HOME first:"
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
  echo "  source \$HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

# Activate conda in the current shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create or update env from environment.yml.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[bootstrap] env exists — updating from environment.yml"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "[bootstrap] creating env from environment.yml"
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

conda activate "${ENV_NAME}"

# Double-install pip requirements to pick up anything environment.yml
# missed. `--upgrade-strategy only-if-needed` keeps the torch build from
# conda intact.
python -m pip install --upgrade pip
python -m pip install --upgrade-strategy only-if-needed -r requirements.txt

# Sanity check: confirm CUDA, DDP prerequisites, and onnxruntime load.
python - <<'PY'
import sys, platform
print("[bootstrap] python:", sys.version.split()[0], platform.machine())
import torch
print("[bootstrap] torch:", torch.__version__, "cuda available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[bootstrap]   gpu{i}: {torch.cuda.get_device_name(i)}")
import timm, pandas, sklearn, scipy, onnx, onnxruntime
print("[bootstrap] timm:", timm.__version__)
print("[bootstrap] onnxruntime providers:", onnxruntime.get_available_providers())
PY

# Prepare the directory tree we'll write into.
mkdir -p results/slurm_logs results/logs_ciai_fast results/checkpoints_ciai_fast results/figures_ciai_fast

# Ensure shell scripts are executable.
chmod +x scripts/*.sh scripts/slurm/*.sbatch 2>/dev/null || true

cat <<EOF

[bootstrap] done.

Next:
  1) Stage HAM10000 raw images at data/ham10000/ (or sync from your Mac).
  2) Build the processed metadata CSV:
       conda activate ${ENV_NAME}
       python -m data.download_ham10000 \\
           --source-dir data/ham10000 \\
           --output-dir data/ham10000
  3) Submit the fast-path jobs:
       sbatch scripts/slurm/job_a_seeds_0_2.sbatch
       sbatch scripts/slurm/job_b_seed_1.sbatch
  4) Watch status:
       bash scripts/cluster_status.sh
EOF
