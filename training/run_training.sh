#!/bin/bash
#
# Run nnUNetv2_plan_and_preprocess, nnUNetv2_train, and nnUNetv2_predict on the dataset
#
# Example usage:
#     bash run_training.sh <GPU> <dataset_id> <dataset_name>
#     bash run_training.sh 1 301 Dataset301_LumbarRootlets
#
# Authors: Naga Karthik, Jan Valosek
#

# !!! MODIFY THE FOLLOWING VARIABLES ACCORDING TO YOUR NEEDS !!!
DEVICE=${1}
dataset_id=${2}                        # e.g. 301
dataset_name=${3}                      # e.g. Dataset301_LumbarRootlets

config="3d_fullres"                     # e.g. 3d_fullres or 2d
nnunet_trainer="nnUNetTrainer"
# default: nnUNetTrainer or nnUNetTrainer_2000epochs
# other options: nnUNetTrainerDA5, nnUNetTrainerDA5_DiceCELoss_noSmooth

# Select number of folds here
# folds=(0 1 2 3 4)
# folds=(0 1 2)
folds=(0)

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"

nnUNetv2_plan_and_preprocess -d ${dataset_id} --verify_dataset_integrity -c ${config}

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=${DEVICE} nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer}

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