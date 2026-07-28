#!/bin/bash
# Launch one or more already-preprocessed Dataset403 folds on Romane.
#
# Invoke this script through set_slot so CUDA_VISIBLE_DEVICES is assigned by
# the cluster wrapper, for example:
#
#   set_slot 0 bash training/cervical_cropped/run_clean_fold_romane.sh "0 4" 0
#   set_slot 1 bash training/cervical_cropped/run_clean_fold_romane.sh "1 all" 1

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || -z "$1" ]]; then
    echo "Usage: $0 \"<space-separated folds>\" [GPU index]" >&2
    exit 2
fi

readonly PROJECT=/home/kuanyiw/projects/rootlets
readonly REPOSITORY="${ROOTLETS_REPO:-${PROJECT}/model-spinal-rootlets}"
readonly DATASET="${ROOTLETS_DATASET:-${PROJECT}/data/Dataset403_CervicalRootletsCroppedCleanT1RPI}"
readonly OUTPUT_ROOT="${ROOTLETS_OUTPUT_ROOT:-${PROJECT}/data/unet_output_cropped_rpi_clean_t1}"
readonly ENV_BIN=/home/kuanyiw/.conda/envs/rootlets-romane/bin
readonly TRAINER=nnUNetTrainer_2000epochsEarlyStopping
readonly DEVICE="${2:-cuda}"

export PATH="${ENV_BIN}:${PATH}"
export nnUNet_raw="${PROJECT}/data"
export nnUNet_preprocessed="${OUTPUT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_ROOT}/nnUNet_results"
export FOLDS="$1"

cd "$REPOSITORY"

"${ENV_BIN}/python" - "$TRAINER" <<'PY'
import sys
from os.path import join

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

trainer_name = sys.argv[1]
trainer = recursive_find_python_class(
    join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
    trainer_name,
    "nnunetv2.training.nnUNetTrainer",
)
if trainer is None:
    raise SystemExit(
        f"{trainer_name} is not installed. Run "
        "training/cervical_cropped/install_early_stopping_trainer_romane.sh first."
    )
PY

exec bash training/run_training.sh \
    "$DEVICE" \
    403 \
    "${DATASET}" \
    3d_fullres \
    "$TRAINER" \
    --train-only
