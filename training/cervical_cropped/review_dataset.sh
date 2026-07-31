#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") /path/to/DatasetXXX_Name" >&2
    echo "" >&2
    echo "Reviews nnU-Net training pairs in imagesTr/ and labelsTr/ with FSLeyes." >&2
}

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]] && exit 0
    exit 2
fi

DATASET_DIR=$1

IMAGE_DIR="$DATASET_DIR/imagesTr"
LABEL_DIR="$DATASET_DIR/labelsTr"

if [[ ! -d "$IMAGE_DIR" || ! -d "$LABEL_DIR" ]]; then
    echo "ERROR: expected both imagesTr/ and labelsTr/ in: $DATASET_DIR" >&2
    exit 2
fi

if ! command -v fsleyes >/dev/null 2>&1; then
    echo "ERROR: fsleyes is not on PATH. Activate the FSL environment first." >&2
    exit 127
fi

# Without nullglob, an empty directory produces one literal '*.nii.gz' item.
shopt -s nullglob
images=("$IMAGE_DIR"/*_0000.nii.gz)
if [[ ${#images[@]} -eq 0 ]]; then
    echo "ERROR: no nnU-Net images matching '*_0000.nii.gz' in: $IMAGE_DIR" >&2
    exit 2
fi

if [[ ! -t 0 ]]; then
    echo "ERROR: this interactive reviewer requires a terminal." >&2
    exit 2
fi

reviewed=0
missing=0
for image in "${images[@]}"; do
    filename="$(basename "$image")"
    case_id="${filename%_0000.nii.gz}"
    label="$LABEL_DIR/${case_id}.nii.gz"

    if [[ ! -f "$label" ]]; then
        echo "Missing label: $label"
        ((missing += 1))
        continue
    fi

    echo
    echo "Case: $case_id"
    read -r -p "[Enter] open | [s] skip | [q] quit: " choice

    case "$choice" in
        q|Q)
            break
            ;;
        s|S)
            continue
            ;;
    esac

    fsleyes \
        "$image" \
        "$label" \
        --overlayType label \
        --alpha 60
    ((reviewed += 1))
done

echo "Reviewed $reviewed case(s); skipped $missing case(s) without a matching label."
