#!/usr/bin/env bash
# Download HAM10000 directly onto the cluster.
#
# Two methods — the script tries them in order:
#
#   1) Harvard Dataverse API (no auth, public dataset)
#      Fetches every file from the HAM10000 dataset deposit at
#      doi:10.7910/DVN/DBW86T. Resolves file IDs dynamically so we
#      don't depend on hardcoded numeric IDs that can rotate.
#
#   2) Kaggle CLI (needs ~/.kaggle/kaggle.json)
#      Fallback path if the Dataverse API is unreachable from the
#      compute network.
#
# After a successful download the script unzips both image parts,
# stages HAM10000_metadata.csv, and prints the next command — the
# metadata-processing step that produces processed_metadata.csv with
# lesion_id so the lesion-grouped splitter works.
#
# Usage (run on a CIAI login node from the repo root):
#   bash scripts/download_ham10000_cluster.sh
#
# Env overrides:
#   DATA_ROOT    target directory (default: data/ham10000)
#   METHOD       force 'dataverse' or 'kaggle'
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/ham10000}"
METHOD="${METHOD:-auto}"
DATAVERSE_DOI="doi:10.7910/DVN/DBW86T"
DATAVERSE_BASE="https://dataverse.harvard.edu"

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

_have() { command -v "$1" >/dev/null 2>&1; }

_dataverse_file_list() {
  # Returns "<id>\t<filename>" lines for every file in the dataset.
  python - <<'PY'
import json, os, sys, urllib.request
doi = "doi:10.7910/DVN/DBW86T"
base = "https://dataverse.harvard.edu"
url = f"{base}/api/datasets/:persistentId/?persistentId={doi}"
try:
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
except Exception as exc:
    sys.stderr.write(f"[dataverse] API request failed: {exc}\n")
    sys.exit(2)
for entry in payload["data"]["latestVersion"]["files"]:
    f = entry["dataFile"]
    print(f"{f['id']}\t{f['filename']}")
PY
}

_dataverse_download() {
  echo "[dataverse] listing files under ${DATAVERSE_DOI} ..."
  local listing
  listing="$(_dataverse_file_list)" || return 2

  while IFS=$'\t' read -r file_id filename; do
    [[ -z "${file_id}" ]] && continue
    if [[ -f "${filename}" ]]; then
      echo "[dataverse] skip (already present): ${filename}"
      continue
    fi
    echo "[dataverse] fetching ${filename} (id=${file_id})"
    # `-L` follows redirects, `-C -` resumes on retry.
    curl -fL -C - -o "${filename}" \
      "${DATAVERSE_BASE}/api/access/datafile/${file_id}"
  done <<< "${listing}"
}

_kaggle_download() {
  if ! _have kaggle; then
    echo "[kaggle] CLI not installed. Inside your conda env: pip install kaggle"
    return 2
  fi
  if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "[kaggle] ~/.kaggle/kaggle.json missing. Generate an API token at"
    echo "          https://www.kaggle.com/settings/account  -> Create New Token"
    return 2
  fi
  echo "[kaggle] downloading kmader/skin-cancer-mnist-ham10000 ..."
  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 --unzip
}

# --- run ---
status=1
if [[ "${METHOD}" == "auto" || "${METHOD}" == "dataverse" ]]; then
  if _dataverse_download; then status=0; fi
fi
if [[ "${status}" -ne 0 && ( "${METHOD}" == "auto" || "${METHOD}" == "kaggle" ) ]]; then
  if _kaggle_download; then status=0; fi
fi
if [[ "${status}" -ne 0 ]]; then
  echo "[error] all download methods failed."
  exit 1
fi

# --- unzip + normalise ---
shopt -s nullglob
for archive in HAM10000_images_part_1.zip HAM10000_images_part_2.zip \
               HAM10000_images.zip; do
  if [[ -f "${archive}" ]]; then
    echo "[unzip] ${archive}"
    unzip -n -q "${archive}"
  fi
done

# Some releases ship metadata as .tab; also accept the Kaggle CSV.
if [[ -f HAM10000_metadata.tab && ! -f HAM10000_metadata.csv ]]; then
  cp HAM10000_metadata.tab HAM10000_metadata.csv
fi

cat <<EOF

[download] done. Directory now has:
$(ls -1 | sed 's/^/    /')

Next step: build the processed metadata CSV that the training pipeline
reads (includes lesion_id, so the lesion-grouped splitter works):

  cd "\${OLDPWD:-..}"
  python -m data.download_ham10000 \\
      --source-dir ${DATA_ROOT} \\
      --output-dir ${DATA_ROOT}

Optional — segmentation masks (W12 attention overlap). HAM10000 does
not ship them; download the ISIC 2018 Task 1 ground truth instead:
  https://challenge.isic-archive.com/data/#2018  -> Task 1 Training GT
Unpack into ${DATA_ROOT}/segmentation_masks/ (filenames must stay
ISIC_<id>_segmentation.png so data.segmentation_mask_dir lookup works).
EOF
