#!/usr/bin/env bash

set -euo pipefail

IMAGE_DIR="${1:-images}"
LABEL_DIR="${2:-labels}"

for image in "$IMAGE_DIR"/*.nii.gz; do
    filename="$(basename "$image")"
    label="$LABEL_DIR/$filename"

    if [[ ! -f "$label" ]]; then
        echo "Skipping $filename: no matching label"
        continue
    fi

    echo
    echo "Reviewing: $filename"
    echo "Close FSLeyes to continue to the next case."

    fsleyes \
        "$image" \
        "$label" \
        --overlayType label \
        --alpha 60
done