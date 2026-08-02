#!/usr/bin/env bash
# Run RootletSeg + deterministic D/V splitting on paired Marseille scans.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DATASET_ROOT="${DATASET_ROOT:-/home/kuanyiw/projects/rootlets/data/marseille-rootlets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/kuanyiw/experiments/dorsal-ventral-v1/marseille}"
CODE_ROOT="${CODE_ROOT:-/home/kuanyiw/experiments/dorsal-ventral-v1/code}"
SCT_DIR="${SCT_DIR:-/home/kuanyiw/spinalcordtoolbox}"
PYTHON_BIN="${PYTHON_BIN:-/home/kuanyiw/.conda/envs/rootlets-romane/bin/python}"
SCT_PYTHON_BIN="${SCT_PYTHON_BIN:-$SCT_DIR/python/envs/venv_sct/bin/python}"
MODE=""
COMPUTE=""
SLOT=""
CUDA_DEVICE=""

usage() {
  echo "Usage: $0 --dry-run | --run --compute {cpu,gpu} --slot {0,1,2,3} [--cuda-device N]"
  echo "For --run, first book the matching resource and set RESOURCE_SLOT_BOOKED=1."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--run|--worker)
      MODE="$1"
      shift
      ;;
    --slot)
      SLOT="${2:?missing value for --slot}"
      shift 2
      ;;
    --compute)
      COMPUTE="${2:?missing value for --compute}"
      shift 2
      ;;
    --cuda-device)
      CUDA_DEVICE="${2:?missing value for --cuda-device}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" == "--run" ]]; then
  if [[ "${RESOURCE_SLOT_BOOKED:-${GPU_SLOT_BOOKED:-0}}" != "1" ]]; then
    echo "Refusing inference: book the resource, then set RESOURCE_SLOT_BOOKED=1." >&2
    exit 2
  fi
  if [[ ! "$SLOT" =~ ^[0-3]$ || ! "$COMPUTE" =~ ^(cpu|gpu)$ ]]; then
    echo "--run requires a valid --slot and --compute {cpu,gpu}." >&2
    exit 2
  fi
  if [[ "$COMPUTE" == "gpu" ]]; then
    if [[ ! "$CUDA_DEVICE" =~ ^[0-3]$ || "$SLOT" != "$CUDA_DEVICE" ]]; then
      echo "GPU mode requires matching --slot and --cuda-device values from 0 to 3." >&2
      exit 2
    fi
    exec set_slot "$SLOT" env \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      SCT_USE_GPU=1 \
      DVSPLIT_IN_SLOT=1 \
      DATASET_ROOT="$DATASET_ROOT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      CODE_ROOT="$CODE_ROOT" \
      SCT_DIR="$SCT_DIR" \
      PYTHON_BIN="$PYTHON_BIN" \
      SCT_PYTHON_BIN="$SCT_PYTHON_BIN" \
      "$SCRIPT_PATH" --worker --compute gpu --slot "$SLOT" --cuda-device "$CUDA_DEVICE"
  fi
  if [[ -n "$CUDA_DEVICE" ]]; then
    echo "CPU mode does not accept --cuda-device." >&2
    exit 2
  fi
  exec set_slot "$SLOT" env -u CUDA_VISIBLE_DEVICES -u SCT_USE_GPU \
    DVSPLIT_IN_SLOT=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CODE_ROOT="$CODE_ROOT" \
    SCT_DIR="$SCT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    SCT_PYTHON_BIN="$SCT_PYTHON_BIN" \
    "$SCRIPT_PATH" --worker --compute cpu --slot "$SLOT"
fi

if [[ "$MODE" != "--dry-run" && "$MODE" != "--worker" ]]; then
  usage >&2
  exit 2
fi
if [[ "$MODE" == "--worker" && "${DVSPLIT_IN_SLOT:-0}" != "1" ]]; then
  echo "Refusing worker mode outside set_slot." >&2
  exit 2
fi
if [[ "$MODE" == "--worker" && ! "$COMPUTE" =~ ^(cpu|gpu)$ ]]; then
  echo "Worker requires --compute {cpu,gpu}." >&2
  exit 2
fi

for required in "$DATASET_ROOT" "$CODE_ROOT" "$SCT_DIR"; do
  if [[ ! -d "$required" ]]; then
    echo "Required directory not found: $required" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$MODE" == "--worker" && "$COMPUTE" == "gpu" ]]; then
  if [[ ! -x "$SCT_PYTHON_BIN" ]] || ! "$SCT_PYTHON_BIN" -c \
    'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "Refusing silent CPU fallback: SCT's Python environment has no usable CUDA." >&2
    exit 1
  fi
  echo "Compute preflight: SCT CUDA is available on device $CUDA_DEVICE."
elif [[ "$MODE" == "--worker" ]]; then
  echo "Compute preflight: explicit CPU mode in slot $SLOT."
fi

INPUTS=()
while IFS= read -r input; do
  INPUTS+=("$input")
done < <(find "$DATASET_ROOT" -type f -name '*_T2w.nii.gz' | sort)
if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "No T2w inputs found under $DATASET_ROOT" >&2
  exit 1
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only: ${#INPUTS[@]} inputs; no GPU, inference, or writes."
  echo "Dataset: $DATASET_ROOT"
  echo "Output: $OUTPUT_ROOT"
  printf '%s\n' "${INPUTS[@]}"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
MANIFEST_TMP="$(mktemp "$OUTPUT_ROOT/manifest.csv.tmp.XXXXXX")"
trap 'rm -f "$MANIFEST_TMP"' EXIT
echo 'subject,session,compute,image,cord,combined,dorsal,ventral,qc,log' > "$MANIFEST_TMP"

validate_partition() {
  "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import sys
import nibabel as nib
import numpy as np

combined_image, dorsal_image, ventral_image = [nib.load(path) for path in sys.argv[1:]]
if any(image.shape != combined_image.shape for image in (dorsal_image, ventral_image)):
    raise SystemExit("output shape mismatch")
if any(not np.allclose(image.affine, combined_image.affine, atol=1e-4) for image in (dorsal_image, ventral_image)):
    raise SystemExit("output affine mismatch")
combined, dorsal, ventral = [np.rint(np.asanyarray(image.dataobj)).astype(np.int32) for image in (combined_image, dorsal_image, ventral_image)]
if np.any((dorsal > 0) & (ventral > 0)) or not np.array_equal(dorsal + ventral, combined):
    raise SystemExit("D/V outputs do not exactly partition RootletSeg support")
PY
}

for input in "${INPUTS[@]}"; do
  filename="$(basename "$input")"
  prefix="${filename%_T2w.nii.gz}"
  subject="${prefix%%_*}"
  session_part="${prefix#*_}"
  session="${session_part%%_*}"
  relative_directory="$(dirname "${input#"$DATASET_ROOT"/}")"
  scan_output="$OUTPUT_ROOT/$relative_directory"
  mkdir -p "$scan_output"

  cord="$scan_output/${prefix}_label-SC_seg.nii.gz"
  combined="$scan_output/${prefix}_label-rootlets_dseg.nii.gz"
  dorsal="$scan_output/${prefix}_desc-dorsal_label-rootlets_dseg.nii.gz"
  ventral="$scan_output/${prefix}_desc-ventral_label-rootlets_dseg.nii.gz"
  score="$scan_output/${prefix}_desc-dvscore.nii.gz"
  qc="$scan_output/${prefix}_desc-dvsplit_qc.json"
  log="$scan_output/${prefix}_dorsal-ventral.log"

  if [[ ! -s "$cord" ]]; then
    "$SCT_DIR/bin/sct_deepseg" spinalcord -i "$input" -o "$cord" 2>&1 | tee -a "$log"
  fi
  if [[ ! -s "$combined" ]]; then
    "$SCT_DIR/bin/sct_deepseg" rootlets -i "$input" -o "$combined" 2>&1 | tee -a "$log"
  fi
  if [[ ! -s "$dorsal" || ! -s "$ventral" || ! -s "$score" || ! -s "$qc" ]]; then
    (
      cd "$CODE_ROOT"
      "$PYTHON_BIN" -m postprocessing.dorsal_ventral.split_rootlets \
        --rootlets "$combined" \
        --cord "$cord" \
        --output-dorsal "$dorsal" \
        --output-ventral "$ventral" \
        --output-score "$score" \
        --qc-json "$qc"
    ) 2>&1 | tee -a "$log"
  fi
  validate_partition "$combined" "$dorsal" "$ventral"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$subject" "$session" "$COMPUTE" "$input" "$cord" "$combined" "$dorsal" "$ventral" "$qc" "$log" \
    >> "$MANIFEST_TMP"
done

mv "$MANIFEST_TMP" "$OUTPUT_ROOT/manifest.csv"
trap - EXIT
(
  cd "$CODE_ROOT"
  "$PYTHON_BIN" -m postprocessing.dorsal_ventral.evaluate_session_consistency \
    --manifest "$OUTPUT_ROOT/manifest.csv" \
    --output-scan-csv "$OUTPUT_ROOT/session_scan_metrics.csv" \
    --output-pair-csv "$OUTPUT_ROOT/session_pair_metrics.csv" \
    --output-json "$OUTPUT_ROOT/session_consistency_summary.json"
  "$PYTHON_BIN" -m postprocessing.dorsal_ventral.render_session_qc \
    --manifest "$OUTPUT_ROOT/manifest.csv" \
    --output-dir "$OUTPUT_ROOT/session_qc"
  "$PYTHON_BIN" -m postprocessing.dorsal_ventral.summarize_inference_runtime \
    --manifest "$OUTPUT_ROOT/manifest.csv" \
    --output-csv "$OUTPUT_ROOT/inference_runtime.csv" \
    --output-json "$OUTPUT_ROOT/inference_runtime_summary.json"
)
