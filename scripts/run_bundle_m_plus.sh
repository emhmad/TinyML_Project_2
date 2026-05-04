#!/usr/bin/env bash
# Bundle M++ orchestrator — submits the full limitations + mechanism
# sweep on the CSCC cluster with the right SLURM dependencies.
#
# Topology:
#
#     P0.1 (login-node diagnostic) ──┐
#                                    │
#     S0 (n=10 sweep) ────────────── ┬──→ L4 (attention IoU)
#                                    └──→ P1 (mechanism: P1.1 + P1.3)
#
#     L2A (seeds 0..2) ──→ L2B (seeds 3..5) ──→ L2C (seeds 6..9)
#     L3A (MobileNetV2) ─→ L3B (ResNet-50)
#
# (L2A/B/C and L3A/B are chained via afterok to dodge PreemptMode=REQUEUE
#  on long jobs.)
#
# Each `sbatch` is submitted with `--parsable` to capture its job id
# and chain the dependent jobs via `--dependency=afterok:<id>`.
#
# Usage:
#   bash scripts/run_bundle_m_plus.sh                # submit everything
#   bash scripts/run_bundle_m_plus.sh --dry-run      # print but don't submit
#   bash scripts/run_bundle_m_plus.sh --skip-l2      # skip ISIC 2019 (e.g. dataset not staged)
#   bash scripts/run_bundle_m_plus.sh --skip-l4      # skip W12 (e.g. seg masks not staged)
#
# Honours these env vars:
#   ENV_NAME    conda env name (default: tinyml-cancer-forget)
#   PARTITION   override SLURM partition (default: from .sbatch headers)

set -euo pipefail

DRY_RUN=0
SKIP_L2=0
SKIP_L3=0
SKIP_L4=0
SKIP_P1=0
SKIP_P0=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-l2) SKIP_L2=1; shift ;;
    --skip-l3) SKIP_L3=1; shift ;;
    --skip-l4) SKIP_L4=1; shift ;;
    --skip-p1) SKIP_P1=1; shift ;;
    --skip-p0) SKIP_P0=1; shift ;;
    -h|--help) head -30 "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

_submit() {
  local script="$1"
  shift
  local extra=("$@")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] sbatch ${extra[*]} ${script}"
    echo "0"
    return
  fi
  sbatch --parsable "${extra[@]}" "${script}"
}

echo "[bundle_m_plus] repo=${REPO_ROOT}"
echo "[bundle_m_plus] dry_run=${DRY_RUN}  skip_l2=${SKIP_L2}  skip_l3=${SKIP_L3}  skip_l4=${SKIP_L4}  skip_p1=${SKIP_P1}"

# ------------------------------------------------------------------
# P0.1 — recovery diagnostic. Runs on the login node, no SLURM job.
# ------------------------------------------------------------------
if [[ "${SKIP_P0}" -eq 0 ]]; then
  echo "[bundle_m_plus] P0.1: diagnosing recovery sweep on existing seed_0 checkpoints"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    python -m scripts.diagnose_recovery \
        --checkpoints-dir results/checkpoints_personal/seed_0 \
        --models deit_small deit_tiny \
        --criteria magnitude wanda taylor \
        --sparsity 0.5 \
        --epochs 5 10 20 \
        --lr 1.0e-4 \
        --output results/logs_personal/seed_0/recovery_diagnostic.csv \
        || echo "[bundle_m_plus] P0.1 diagnostic FAILED — investigate before continuing"
  else
    echo "[dry-run] would run scripts.diagnose_recovery"
  fi
fi

# ------------------------------------------------------------------
# S0 — n=10 pruning matrix. Most other items depend on this.
# ------------------------------------------------------------------
echo "[bundle_m_plus] S0: submitting pruning matrix at n=10 (seeds 3..9)"
S0_JOB=$(_submit scripts/slurm/job_s0_seeds_3to9.sbatch)
echo "[bundle_m_plus]   S0 job id: ${S0_JOB}"

# ------------------------------------------------------------------
# L2 — ISIC 2019 retrained sweep, split into 3 chained chunks to dodge
# preemption. Each chunk is short enough that a requeue costs at most
# ~16h, vs ~48h for the original mega-job.
#   chunk_a: seeds {0, 1, 2}  — runs the dense fine-tune for seed 0
#   chunk_b: seeds {3, 4, 5}  — depends on chunk_a (needs seed-0 ckpt)
#   chunk_c: seeds {6, 7, 8, 9} — depends on chunk_b
# ------------------------------------------------------------------
if [[ "${SKIP_L2}" -eq 0 ]]; then
  if [[ ! -f data/isic2019/processed_metadata.csv ]]; then
    echo "[bundle_m_plus] L2: SKIPPING — data/isic2019/processed_metadata.csv missing."
    echo "                Build it with: python -m scripts.build_isic2019_metadata \\"
    echo "                                  --source-dir data/isic2019 --output-dir data/isic2019"
    SKIP_L2=1
  else
    echo "[bundle_m_plus] L2: submitting ISIC 2019 chunk A (seeds 0,1,2)"
    L2A_JOB=$(_submit scripts/slurm/job_l2_chunk_a.sbatch)
    echo "[bundle_m_plus]   L2A job id: ${L2A_JOB}"

    L2B_DEP=()
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      L2B_DEP=(--dependency=afterok:"${L2A_JOB}")
    fi
    echo "[bundle_m_plus] L2: submitting ISIC 2019 chunk B (seeds 3,4,5; afterok L2A)"
    L2B_JOB=$(_submit scripts/slurm/job_l2_chunk_b.sbatch "${L2B_DEP[@]}")
    echo "[bundle_m_plus]   L2B job id: ${L2B_JOB}"

    L2C_DEP=()
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      L2C_DEP=(--dependency=afterok:"${L2B_JOB}")
    fi
    echo "[bundle_m_plus] L2: submitting ISIC 2019 chunk C (seeds 6,7,8,9; afterok L2B)"
    L2C_JOB=$(_submit scripts/slurm/job_l2_chunk_c.sbatch "${L2C_DEP[@]}")
    echo "[bundle_m_plus]   L2C job id: ${L2C_JOB}"
  fi
fi

# ------------------------------------------------------------------
# L3 — CNN baselines, split into 2 chained chunks by architecture.
#   chunk_a: pillar 7 (MobileNetV2)
#   chunk_b: pillar 8 (ResNet-50) — depends on chunk_a only for resource
#                                   serialisation; technically independent
#                                   since separate checkpoints/results dirs.
# ------------------------------------------------------------------
if [[ "${SKIP_L3}" -eq 0 ]]; then
  echo "[bundle_m_plus] L3: submitting MobileNetV2 chunk A"
  L3A_JOB=$(_submit scripts/slurm/job_l3_chunk_a.sbatch)
  echo "[bundle_m_plus]   L3A job id: ${L3A_JOB}"

  L3B_DEP=()
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    L3B_DEP=(--dependency=afterok:"${L3A_JOB}")
  fi
  echo "[bundle_m_plus] L3: submitting ResNet-50 chunk B (afterok L3A)"
  L3B_JOB=$(_submit scripts/slurm/job_l3_chunk_b.sbatch "${L3B_DEP[@]}")
  echo "[bundle_m_plus]   L3B job id: ${L3B_JOB}"
fi

# ------------------------------------------------------------------
# L4 — quantitative attention IoU. Depends on S0 (needs n=10 masks).
# ------------------------------------------------------------------
if [[ "${SKIP_L4}" -eq 0 ]]; then
  if [[ ! -d data/ham10000/segmentation_masks ]] || [[ -z "$(ls -A data/ham10000/segmentation_masks 2>/dev/null || true)" ]]; then
    echo "[bundle_m_plus] L4: SKIPPING — segmentation masks not staged."
    echo "                Build via: python -m scripts.build_isic2018_segmasks \\"
    echo "                                --source-dir <ISIC2018_GroundTruth_dir> \\"
    echo "                                --metadata-csv data/ham10000/processed_metadata.csv \\"
    echo "                                --output-dir data/ham10000/segmentation_masks"
    SKIP_L4=1
  else
    echo "[bundle_m_plus] L4: submitting attention IoU (depends on S0=${S0_JOB})"
    L4_DEP=()
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      L4_DEP=(--dependency=afterok:"${S0_JOB}")
    fi
    L4_JOB=$(_submit scripts/slurm/job_l4_attention_iou.sbatch "${L4_DEP[@]}")
    echo "[bundle_m_plus]   L4 job id: ${L4_JOB}"
  fi
fi

# ------------------------------------------------------------------
# P1 — mechanism probes (P1.1 + P1.3). Depends on S0.
# ------------------------------------------------------------------
if [[ "${SKIP_P1}" -eq 0 ]]; then
  echo "[bundle_m_plus] P1: submitting mechanism probes (P1.1 + P1.3) (depends on S0=${S0_JOB})"
  P1_DEP=()
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    P1_DEP=(--dependency=afterok:"${S0_JOB}")
  fi
  P1_JOB=$(_submit scripts/slurm/job_p1_mechanism.sbatch "${P1_DEP[@]}")
  echo "[bundle_m_plus]   P1 job id: ${P1_JOB}"
fi

echo
echo "[bundle_m_plus] all submissions issued. Watch progress with:"
echo "    bash scripts/cluster_status.sh"
echo "    squeue -u \$USER"
