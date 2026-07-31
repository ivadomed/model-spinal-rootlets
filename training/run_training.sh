#!/bin/bash
#
# Run nnUNetv2_plan_and_preprocess, nnUNetv2_train, and nnUNetv2_predict on the dataset
#
# Example usage:
#     bash run_training.sh <device> <dataset_id> <dataset_dir_or_name> <config> <trainer> [--train-only|--continue]
#     bash run_training.sh 1 301 /data/nnUNet_raw/Dataset301_LumbarRootlets 3d_fullres nnUNetTrainer
#     bash run_training.sh mps 301 /data/nnUNet_raw/Dataset301_LumbarRootlets 3d_fullres nnUNetTrainer
#     bash run_training.sh 0 401 /data/Dataset401_CervicalRootletsCropped 3d_fullres nnUNetTrainer_2000epochs --train-only
#     bash run_training.sh 0 401 /data/Dataset401_CervicalRootletsCropped 3d_fullres nnUNetTrainer_250epochs --continue
#
# Authors: Naga Karthik, Jan Valosek
#

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: $0 <device|GPU index> <dataset_id> <dataset_dir_or_name> <config> <trainer> [--train-only|--continue]" >&2
    echo "  device: mps, cpu, cuda, or a CUDA GPU index (for example 0)" >&2
    echo "  --train-only: reuse existing preprocessing and start a new training" >&2
    echo "  --continue: skip preprocessing and resume each fold from checkpoint_latest.pth" >&2
    exit 2
fi

DEVICE=$1                              # mps, cpu, cuda, or a CUDA GPU index
dataset_id=$2                          # e.g. 301
dataset_arg=$3                         # path or name, e.g. /data/nnUNet_raw/Dataset301_LumbarRootlets
config=$4                              # e.g. 3d_fullres or 2d
nnunet_trainer=$5                      # default: nnUNetTrainer
                                       # other options: nnUNetTrainer_250epochs, nnUNetTrainer_2000epochs,
                                       # nnUNetTrainerDA5, nnUNetTrainerDA5_DiceCELoss_noSmooth
run_mode=${6:-}

if [[ -n "$run_mode" && "$run_mode" != "--train-only" && "$run_mode" != "--continue" ]]; then
    echo "Invalid optional argument '$run_mode'. Use --train-only or --continue." >&2
    exit 2
fi

# nnU-Net v2 discovers datasets exclusively through these three variables. When
# a dataset path is supplied, infer a self-contained layout beside that dataset.
# Explicitly exported values still take precedence for preprocessed data/results.
if [[ -d "$dataset_arg" ]]; then
    dataset_dir=$(cd "$dataset_arg" && pwd)
    dataset_name=$(basename "$dataset_dir")
    export nnUNet_raw=$(dirname "$dataset_dir")
    export nnUNet_preprocessed=${nnUNet_preprocessed:-"${nnUNet_raw}/nnUNet_preprocessed"}
    export nnUNet_results=${nnUNet_results:-"${nnUNet_raw}/nnUNet_results"}
else
    dataset_name=$dataset_arg
    if [[ -z "${nnUNet_raw:-}" ]]; then
        echo "Dataset directory '$dataset_arg' does not exist and nnUNet_raw is not set." >&2
        echo "Pass the full dataset directory, or export nnUNet_raw first." >&2
        exit 1
    fi
    export nnUNet_preprocessed=${nnUNet_preprocessed:-"${nnUNet_raw}/nnUNet_preprocessed"}
    export nnUNet_results=${nnUNet_results:-"${nnUNet_raw}/nnUNet_results"}
    dataset_dir="${nnUNet_raw}/${dataset_name}"
fi

if [[ ! -f "${dataset_dir}/dataset.json" ]]; then
    echo "Missing required nnU-Net metadata: ${dataset_dir}/dataset.json" >&2
    exit 1
fi

if [[ ! "$dataset_name" =~ ^Dataset0*${dataset_id}(_|$) ]]; then
    echo "Dataset ID ${dataset_id} does not match directory name '${dataset_name}'." >&2
    exit 1
fi

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

# nnU-Net selects the backend with -device. Preserve the original behavior of
# accepting a numeric CUDA GPU index, while also supporting Apple MPS and CPU.
case "$DEVICE" in
    mps)
        train_device="mps"
        # Let unsupported MPS operations fall back to CPU where PyTorch permits.
        export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
        ;;
    cpu)
        train_device="cpu"
        ;;
    cuda)
        train_device="cuda"
        ;;
    ''|*[!0-9]*)
        echo "Invalid device '$DEVICE'. Use mps, cpu, cuda, or a CUDA GPU index." >&2
        exit 2
        ;;
    *)
        train_device="cuda"
        export CUDA_VISIBLE_DEVICES="$DEVICE"
        ;;
esac

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "training_device=$train_device"

# Check whether config is valid, if not, exit
if [[ ${config} != "2d" && ${config} != "3d_fullres" ]]; then
    echo "Invalid configuration. Please use either 2d or 3d_fullres."
    exit 1
fi

# Check whether nnunet_trainer is valid, if not, exit
available_trainers=("nnUNetTrainer" "nnUNetTrainer_250epochs" "nnUNetTrainer_2000epochs" "nnUNetTrainer_2000epochsEarlyStopping" "nnUNetTrainerDA5" "nnUNetTrainerDA5_DiceCELoss_noSmooth")
if [[ ! " ${available_trainers[@]} " =~ " ${nnunet_trainer} " ]]; then
    echo "Invalid nnUNet trainer. Please use one of the following: ${available_trainers[@]}"
    exit 1
fi

# Select number of folds here
# folds=(0 1 2 3 4)
# folds=(0 1 2)
read -r -a folds <<< "${FOLDS:-0}"
for fold in "${folds[@]}"; do
    if [[ ! "$fold" =~ ^([0-4]|all)$ ]]; then
        echo "Invalid fold '$fold'. Set FOLDS to a space-separated subset of: 0 1 2 3 4 all" >&2
        exit 2
    fi
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "$run_mode" == "--train-only" || "$run_mode" == "--continue" ]]; then
    if [[ "$run_mode" == "--continue" ]]; then
        echo "Skipping preprocessing; continuing training from existing checkpoints."
    else
        echo "Skipping preprocessing; starting a new training with existing preprocessed data."
    fi
    if [[ ! -d "${nnUNet_preprocessed}/${dataset_name}" ]]; then
        echo "Missing preprocessed dataset: ${nnUNet_preprocessed}/${dataset_name}" >&2
        exit 1
    fi
else
    echo "-------------------------------------------------------"
    echo "Running preprocessing and verifying dataset integrity"
    echo "-------------------------------------------------------"

    nnUNetv2_plan_and_preprocess -d "$dataset_id" --verify_dataset_integrity -c "$config"

    # Preserve the published train/validation/test assignments for the cropped
    # cervical-rootlets dataset. nnU-Net does not read the CSV itself; it consumes
    # splits_final.json from the preprocessed dataset directory.
    if [[ "$dataset_name" == *CervicalRootletsCropped* ]]; then
        split_csv="${script_dir}/cervical_cropped/MP2RAGE_T2w_fold_splits.csv"
        split_script="${script_dir}/cervical_cropped/create_splits.py"
        split_output="${nnUNet_preprocessed}/${dataset_name}/splits_final.json"
        if [[ ! -f "$split_csv" || ! -f "$split_script" ]]; then
            echo "Missing cervical-rootlets split tooling under ${script_dir}/cervical_cropped." >&2
            exit 1
        fi
        python "$split_script" --dataset "$dataset_dir" --csv "$split_csv" --output "$split_output"
    fi
fi

for fold in "${folds[@]}"; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    continue_args=()
    if [[ "$run_mode" == "--continue" ]]; then
        continue_args=(--c)
    fi
    nnUNetv2_train "$dataset_id" "$config" "$fold" -tr "$nnunet_trainer" -device "$train_device" "${continue_args[@]}"

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

#    # inference
#    CUDA_VISIBLE_DEVICES=${DEVICE} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${nnunet_trainer}__nnUNetPlans__${config}/fold_${fold}/test -d ${dataset_id} -f ${fold} -c ${config}
#
#    echo ""
#    echo "-------------------------------------------"
#    echo " Inference completed on Fold $fold"
#    echo "-------------------------------------------"

done
