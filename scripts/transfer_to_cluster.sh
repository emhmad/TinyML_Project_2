#!/usr/bin/env bash
# Rsync the project to the CIAI cluster from your MacBook (W14-style
# release workflow, local side).
#
# Usage:
#   bash scripts/transfer_to_cluster.sh [remote_user] [remote_host] [remote_path]
#
# Defaults to Kaiser's CIAI account. Uses rsync over ssh with --delete
# disabled (safer — won't nuke in-progress cluster results if your
# local copy is stale).
#
# Excluded:
#   - .git history          (fetch on the cluster if needed)
#   - __pycache__ / *.pyc
#   - results/ + checkpoints (rebuilt on the cluster)
#   - data/ham10000 raw images (send separately — huge, see tail of file)
set -euo pipefail

REMOTE_USER="${1:-abdulla.alfalasi}"
REMOTE_HOST="${2:-ciai.mbzuai.ac.ae}"
REMOTE_PATH="${3:-~/TinyML_Project_2}"

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[transfer] local:  ${LOCAL_ROOT}"
echo "[transfer] remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# Apple's stock rsync (macOS) is 2.6.9 and doesn't know --info.
# Detect once and pick a progress flag that works on both.
if rsync --info=progress2 --version >/dev/null 2>&1; then
  PROGRESS_FLAG=(--info=progress2)
else
  PROGRESS_FLAG=(--progress)
  echo "[transfer] using legacy rsync --progress (consider 'brew install rsync' for a modern build)"
fi

# --safe-links is also rsync-3; on stock macOS we skip it quietly.
if rsync --safe-links --version >/dev/null 2>&1; then
  SAFE_LINKS_FLAG=(--safe-links)
else
  SAFE_LINKS_FLAG=()
fi

rsync -avz \
  "${PROGRESS_FLAG[@]}" \
  "${SAFE_LINKS_FLAG[@]}" \
  --human-readable \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'results/' \
  --exclude 'data/ham10000/' \
  --exclude 'release/' \
  -e 'ssh' \
  "${LOCAL_ROOT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

cat <<'EOF'

[transfer] code copied. Next steps on the cluster:

  ssh aalbudoor@ciai.mbzuai.ac.ae
  cd ~/TinyML_Project_2
  bash scripts/cluster_bootstrap.sh

To push the HAM10000 raw images separately (only if you've already
downloaded them locally):

  rsync -avz --progress \
    data/ham10000/ \
    aalbudoor@ciai.mbzuai.ac.ae:~/TinyML_Project_2/data/ham10000/

Or pull them directly on the cluster via ISIC/Kaggle (see
data/download_ham10000.py for the expected directory layout).
EOF
