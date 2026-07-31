#!/bin/bash
# Validate completed Dataset403 folds with checkpoint_best in a private result copy.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <evaluation-output-root> <fold> [fold ...]" >&2
    exit 2
fi

readonly OUTPUT_ROOT=$1
shift
readonly FOLDS=("$@")
readonly PROJECT=/home/kuanyiw/projects/rootlets
readonly ENV_BIN=/home/kuanyiw/.conda/envs/rootlets-romane/bin
readonly DATASET=Dataset403_CervicalRootletsCroppedCleanT1RPI
readonly TRAINER=nnUNetTrainer_2000epochsEarlyStopping
readonly CONFIGURATION=3d_fullres
readonly TRAINER_ROOT="${OUTPUT_ROOT}/nnUNet_results/${DATASET}/${TRAINER}__nnUNetPlans__${CONFIGURATION}"

export PATH="${ENV_BIN}:${PATH}"
export nnUNet_raw="${PROJECT}/data"
export nnUNet_preprocessed="${ROOTLETS_PREPROCESSED:-${PROJECT}/data/unet_output_cropped_rpi_clean_t1/nnUNet_preprocessed}"
export nnUNet_results="${OUTPUT_ROOT}/nnUNet_results"

for fold in "${FOLDS[@]}"; do
    if [[ ! "$fold" =~ ^[0-4]$ ]]; then
        echo "Invalid fold: $fold" >&2
        exit 2
    fi

    fold_root="${TRAINER_ROOT}/fold_${fold}"
    if [[ ! -f "${fold_root}/checkpoint_best.pth" ]]; then
        echo "Missing best checkpoint: ${fold_root}/checkpoint_best.pth" >&2
        exit 1
    fi
    if [[ -e "${fold_root}/validation" || -e "${fold_root}/validation_best" ]]; then
        echo "Refusing to replace an existing best-checkpoint validation for fold ${fold}." >&2
        exit 1
    fi

    nnUNetv2_train \
        403 \
        "$CONFIGURATION" \
        "$fold" \
        -tr "$TRAINER" \
        --val \
        --val_best \
        -device cuda

    if [[ ! -f "${fold_root}/validation/summary.json" ]]; then
        echo "Fold ${fold} did not produce validation/summary.json" >&2
        exit 1
    fi
    mv "${fold_root}/validation" "${fold_root}/validation_best"
done
