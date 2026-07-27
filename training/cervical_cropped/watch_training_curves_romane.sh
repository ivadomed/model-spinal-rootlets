#!/bin/bash
# Periodically refresh per-class curves for Dataset403 until all six runs finish.

set -u

readonly RESULTS_ROOT="${RESULTS_ROOT:-/home/kuanyiw/projects/rootlets/data/unet_output_cropped_rpi_clean_t1/nnUNet_results/Dataset403_CervicalRootletsCroppedCleanT1RPI/nnUNetTrainer_2000epochsEarlyStopping__nnUNetPlans__3d_fullres}"
readonly PYTHON="${CURVE_PYTHON:-/home/kuanyiw/experiments/logs/rootlets-clean-20260726/curves/venv/bin/python}"
readonly ROOTLETS_REPO="${ROOTLETS_REPO:-/home/kuanyiw/projects/rootlets/model-spinal-rootlets}"
readonly PLOTTER="${CURVE_PLOTTER:-${ROOTLETS_REPO}/training/cervical_cropped/plot_nnunet_training_log.py}"
readonly INTERVAL="${CURVE_INTERVAL_SECONDS:-1800}"

refresh_curves() {
    while IFS= read -r -d '' log_file; do
        if grep -q "Pseudo dice" "$log_file"; then
            "$PYTHON" "$PLOTTER" -i "$log_file" || true
        fi
    done < <(find "$RESULTS_ROOT" -type f -name "training_log_*.txt" -print0)
}

all_runs_finished() {
    local fold
    for fold in 0 1 2 3 4 all; do
        if [[ ! -f "${RESULTS_ROOT}/fold_${fold}/checkpoint_final.pth" ]]; then
            return 1
        fi
    done
}

while true; do
    refresh_curves
    if all_runs_finished; then
        exit 0
    fi
    sleep "$INTERVAL"
done
