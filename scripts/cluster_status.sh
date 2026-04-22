#!/usr/bin/env bash
# One-shot status view for the running sweep.
#
# Shows:
#   - Your queue position / running state (squeue)
#   - GPU utilisation for the current running job (if any)
#   - Tail of the most recent SLURM stdout/stderr log
#   - Row counts per results CSV so you can see progress
#
# Usage:
#   bash scripts/cluster_status.sh             # one-shot snapshot
#   WATCH=1 bash scripts/cluster_status.sh     # refresh every 20s
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="${RESULTS_DIR:-results/logs_ciai_fast}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-results/slurm_logs}"
USER_NAME="${USER:-$(whoami)}"

_render() {
  clear 2>/dev/null || true
  echo "=== $(date) ==="
  echo
  echo "-- SLURM queue --"
  if command -v squeue >/dev/null 2>&1; then
    squeue -u "${USER_NAME}" -o '%.12i %.9P %.16j %.8u %.2t %.10M %.6D %R' || true
  else
    echo "squeue not on PATH (not a SLURM login node?)"
  fi
  echo

  echo "-- most recent SLURM stdout tails --"
  if [[ -d "${SLURM_LOG_DIR}" ]]; then
    # Show the 20 most recent lines from each of the 2 newest .out files.
    mapfile -t LATEST_OUTS < <(ls -1t "${SLURM_LOG_DIR}"/*.out 2>/dev/null | head -n 2)
    for f in "${LATEST_OUTS[@]}"; do
      echo "--- ${f} ---"
      tail -n 20 "${f}" 2>/dev/null || true
      echo
    done
  else
    echo "no ${SLURM_LOG_DIR} yet"
  fi
  echo

  echo "-- results CSV row counts --"
  if [[ -d "${RESULTS_DIR}" ]]; then
    for csv in "${RESULTS_DIR}"/*.csv "${RESULTS_DIR}"/seed_*/*.csv; do
      [[ -f "${csv}" ]] || continue
      rows=$(( $(wc -l < "${csv}") - 1 ))
      printf "  %-60s %s rows\n" "${csv#${REPO_ROOT}/}" "${rows}"
    done
  else
    echo "no ${RESULTS_DIR} yet"
  fi
  echo

  echo "-- GPU utilisation (running jobs only) --"
  RUNNING_JOBS=$(squeue -u "${USER_NAME}" -h -t R -o '%i' 2>/dev/null || true)
  if [[ -n "${RUNNING_JOBS}" ]]; then
    for jid in ${RUNNING_JOBS}; do
      NODE=$(squeue -j "${jid}" -h -o '%N' 2>/dev/null || true)
      echo "--- job ${jid} on ${NODE} ---"
      srun --jobid="${jid}" --overlap nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null || echo "  (unable to attach with srun --overlap; skipping)"
    done
  else
    echo "no running jobs"
  fi
}

if [[ "${WATCH:-0}" == "1" ]]; then
  while true; do
    _render
    sleep 20
  done
else
  _render
fi
