#!/usr/bin/env bash
# Run the cropped RootletSeg test set and deterministic side-paired D/V split on Romane.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DATASET_ROOT="${DATASET_ROOT:-/home/kuanyiw/projects/rootlets/data/Dataset403_CervicalRootletsCroppedCleanT1RPI}"
NNUNET_RESULTS_ROOT="${NNUNET_RESULTS_ROOT:-/home/kuanyiw/projects/rootlets/data/unet_output_cropped_rpi_clean_t1_completed_fold_metrics/nnUNet_results}"
MODEL_DATASET="${MODEL_DATASET:-Dataset403_CervicalRootletsCroppedCleanT1RPI}"
MODEL_TRAINER="${MODEL_TRAINER:-nnUNetTrainer_2000epochsEarlyStopping}"
MODEL_PLANS="${MODEL_PLANS:-nnUNetPlans}"
MODEL_CONFIGURATION="${MODEL_CONFIGURATION:-3d_fullres}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/kuanyiw/experiments/dorsal-ventral-v1/cropped-dataset403-v3}"
CODE_ROOT="${CODE_ROOT:-/home/kuanyiw/experiments/dorsal-ventral-v1/code}"
NNUNET_PYTHON="${NNUNET_PYTHON:-/home/kuanyiw/.conda/envs/rootlets-romane/bin/python}"
NNUNET_PREDICT="${NNUNET_PREDICT:-/home/kuanyiw/.conda/envs/rootlets-romane/bin/nnUNetv2_predict}"
SCT_DIR="${SCT_DIR:-/home/kuanyiw/spinalcordtoolbox}"
POSTPROCESS_PYTHON="${POSTPROCESS_PYTHON:-/home/kuanyiw/.conda/envs/rootlets-romane/bin/python}"
SCT_CORD_WORKERS="${SCT_CORD_WORKERS:-2}"
MODE=""
SLOT=""
CUDA_DEVICE=""

usage() {
  echo "Usage: $0 --dry-run | --run --slot {0,1,2,3} --cuda-device {0,1,2,3}"
  echo "For --run, first book the matching GPU slot and set RESOURCE_SLOT_BOOKED=1."
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
  if [[ "${RESOURCE_SLOT_BOOKED:-0}" != "1" ]]; then
    echo "Refusing inference: book the resource, then set RESOURCE_SLOT_BOOKED=1." >&2
    exit 2
  fi
  if [[ ! "$SLOT" =~ ^[0-3]$ || ! "$CUDA_DEVICE" =~ ^[0-3]$ || "$SLOT" != "$CUDA_DEVICE" ]]; then
    echo "GPU mode requires matching --slot and --cuda-device values from 0 to 3." >&2
    exit 2
  fi
  exec set_slot "$SLOT" env \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    DV_CROPPED_IN_SLOT=1 \
    DATASET_ROOT="$DATASET_ROOT" \
    NNUNET_RESULTS_ROOT="$NNUNET_RESULTS_ROOT" \
    MODEL_DATASET="$MODEL_DATASET" \
    MODEL_TRAINER="$MODEL_TRAINER" \
    MODEL_PLANS="$MODEL_PLANS" \
    MODEL_CONFIGURATION="$MODEL_CONFIGURATION" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CODE_ROOT="$CODE_ROOT" \
    NNUNET_PYTHON="$NNUNET_PYTHON" \
    NNUNET_PREDICT="$NNUNET_PREDICT" \
    SCT_DIR="$SCT_DIR" \
    POSTPROCESS_PYTHON="$POSTPROCESS_PYTHON" \
    SCT_CORD_WORKERS="$SCT_CORD_WORKERS" \
    "$SCRIPT_PATH" --worker --slot "$SLOT" --cuda-device "$CUDA_DEVICE"
fi

if [[ "$MODE" != "--dry-run" && "$MODE" != "--worker" ]]; then
  usage >&2
  exit 2
fi
if [[ "$MODE" == "--worker" && "${DV_CROPPED_IN_SLOT:-0}" != "1" ]]; then
  echo "Refusing worker mode outside set_slot." >&2
  exit 2
fi

INPUT_DIRECTORY="$DATASET_ROOT/imagesTs"
MODEL_DIRECTORY="$NNUNET_RESULTS_ROOT/$MODEL_DATASET/$MODEL_TRAINER""__""$MODEL_PLANS""__""$MODEL_CONFIGURATION"
for required in "$INPUT_DIRECTORY" "$MODEL_DIRECTORY" "$CODE_ROOT" "$SCT_DIR"; do
  if [[ ! -d "$required" ]]; then
    echo "Required directory not found: $required" >&2
    exit 1
  fi
done
for executable in "$NNUNET_PYTHON" "$NNUNET_PREDICT" "$POSTPROCESS_PYTHON" "$SCT_DIR/bin/sct_deepseg"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done

INPUTS=()
while IFS= read -r input; do
  INPUTS+=("$input")
done < <(find "$INPUT_DIRECTORY" -maxdepth 1 -type f -name '*_0000.nii.gz' | sort)
if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "No nnU-Net test inputs found under $INPUT_DIRECTORY" >&2
  exit 1
fi
if [[ ! "$SCT_CORD_WORKERS" =~ ^[1-4]$ ]]; then
  echo "SCT_CORD_WORKERS must be an integer from 1 to 4." >&2
  exit 2
fi

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only: ${#INPUTS[@]} fixed test images; no inference or writes."
  echo "Dataset: $DATASET_ROOT"
  echo "Model: $MODEL_DIRECTORY"
  echo "Output: $OUTPUT_ROOT"
  exit 0
fi

if ! "$NNUNET_PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "Refusing GPU inference: the nnU-Net Python environment has no usable CUDA." >&2
  exit 1
fi

PREDICTION_DIRECTORY="$OUTPUT_ROOT/rootlet_predictions"
CORD_DIRECTORY="$OUTPUT_ROOT/cord"
SPLIT_DIRECTORY="$OUTPUT_ROOT/dv_sidepaired_v3"
mkdir -p "$PREDICTION_DIRECTORY" "$CORD_DIRECTORY" "$SPLIT_DIRECTORY"

env nnUNet_results="$NNUNET_RESULTS_ROOT" \
  "$NNUNET_PREDICT" \
  -i "$INPUT_DIRECTORY" \
  -o "$PREDICTION_DIRECTORY" \
  -d "$MODEL_DATASET" \
  -tr "$MODEL_TRAINER" \
  -p "$MODEL_PLANS" \
  -c "$MODEL_CONFIGURATION" \
  -f 0 1 2 3 4 \
  -chk checkpoint_best.pth \
  -device cuda \
  -npp 1 \
  -nps 1 \
  --disable_progress_bar \
  --continue_prediction

# The cord model is CPU-only on Romane.  Process a small, bounded batch before
# the deterministic D/V split so that a slow crop does not serialize all cases.
active_cord_jobs=0
for input in "${INPUTS[@]}"; do
  filename="$(basename "$input")"
  case_id="${filename%_0000.nii.gz}"
  combined="$PREDICTION_DIRECTORY/$case_id.nii.gz"
  cord="$CORD_DIRECTORY/${case_id}_label-SC_seg.nii.gz"

  if [[ ! -s "$combined" ]]; then
    echo "Missing nnU-Net prediction: $combined" >&2
    exit 1
  fi
  if [[ -s "$cord" ]]; then
    continue
  fi

  echo "Cord segmentation: $case_id"
  "$SCT_DIR/bin/sct_deepseg" spinalcord -i "$input" -o "$cord" &
  active_cord_jobs=$((active_cord_jobs + 1))
  if (( active_cord_jobs >= SCT_CORD_WORKERS )); then
    wait -n
    active_cord_jobs=$((active_cord_jobs - 1))
  fi
done
while (( active_cord_jobs > 0 )); do
  wait -n
  active_cord_jobs=$((active_cord_jobs - 1))
done

MANIFEST_TMP="$(mktemp "$OUTPUT_ROOT/manifest.csv.tmp.XXXXXX")"
trap 'rm -f "$MANIFEST_TMP"' EXIT
echo 'case,image,combined,cord,dorsal,ventral,qc,rootlet_model' > "$MANIFEST_TMP"

validate_partition() {
  "$POSTPROCESS_PYTHON" - "$1" "$2" "$3" <<'PY'
import sys
import nibabel as nib
import numpy as np

combined_image, dorsal_image, ventral_image = [nib.load(path) for path in sys.argv[1:]]
if any(image.shape != combined_image.shape for image in (dorsal_image, ventral_image)):
    raise SystemExit("output shape mismatch")
if any(not np.allclose(image.affine, combined_image.affine, atol=1e-4) for image in (dorsal_image, ventral_image)):
    raise SystemExit("output affine mismatch")
combined, dorsal, ventral = [
    np.rint(np.asanyarray(image.dataobj)).astype(np.int32)
    for image in (combined_image, dorsal_image, ventral_image)
]
if np.any((dorsal > 0) & (ventral > 0)) or not np.array_equal(dorsal + ventral, combined):
    raise SystemExit("D/V outputs do not exactly partition RootletSeg support")
PY
}

for input in "${INPUTS[@]}"; do
  filename="$(basename "$input")"
  case_id="${filename%_0000.nii.gz}"
  combined="$PREDICTION_DIRECTORY/$case_id.nii.gz"
  cord="$CORD_DIRECTORY/${case_id}_label-SC_seg.nii.gz"
  dorsal="$SPLIT_DIRECTORY/${case_id}_desc-dorsal_label-rootlets_dseg.nii.gz"
  ventral="$SPLIT_DIRECTORY/${case_id}_desc-ventral_label-rootlets_dseg.nii.gz"
  score="$SPLIT_DIRECTORY/${case_id}_desc-dvscore.nii.gz"
  qc="$SPLIT_DIRECTORY/${case_id}_desc-dvsplit_qc.json"

  if [[ ! -s "$cord" ]]; then
    echo "Missing spinal-cord prediction: $cord" >&2
    exit 1
  fi
  if [[ ! -s "$dorsal" || ! -s "$ventral" || ! -s "$score" || ! -s "$qc" ]]; then
    (
      cd "$CODE_ROOT"
      "$POSTPROCESS_PYTHON" -m postprocessing.dorsal_ventral.split_rootlets \
        --rootlets "$combined" \
        --cord "$cord" \
        --output-dorsal "$dorsal" \
        --output-ventral "$ventral" \
        --output-score "$score" \
        --qc-json "$qc" \
        --seed-strategy paired_attachment \
        --paired-min-span-mm 0.5
    )
  fi
  validate_partition "$combined" "$dorsal" "$ventral"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$case_id" "$input" "$combined" "$cord" "$dorsal" "$ventral" "$qc" "$MODEL_DIRECTORY" \
    >> "$MANIFEST_TMP"
done

mv "$MANIFEST_TMP" "$OUTPUT_ROOT/manifest.csv"
trap - EXIT
echo "Completed ${#INPUTS[@]} fixed test images: $OUTPUT_ROOT/manifest.csv"
