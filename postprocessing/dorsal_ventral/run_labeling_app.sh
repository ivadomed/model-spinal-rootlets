#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIRECTORY/../.." && pwd)"
APP_PYTHON_BIN="${APP_PYTHON_BIN:-python3}"
APP_PORT="${APP_PORT:-8501}"

if ! "$APP_PYTHON_BIN" -c 'import nibabel, streamlit' >/dev/null 2>&1; then
  echo "Install the labeling dependencies first:" >&2
  echo "  $APP_PYTHON_BIN -m pip install -r $SCRIPT_DIRECTORY/requirements.txt" >&2
  exit 1
fi

cd "$REPOSITORY_ROOT"
exec "$APP_PYTHON_BIN" -m streamlit run \
  postprocessing/dorsal_ventral/label_rootlets_app.py \
  --server.address 127.0.0.1 \
  --server.port "$APP_PORT" \
  --browser.gatherUsageStats false \
  "$@"
