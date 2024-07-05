#!/bin/bash
#
# This script performs:
# - segmentation of rootlets (model model-spinal-rootlets_ventral_D106_r20240523) on the HC-Leipzig 7T MP2RAGE dataset,
# Authors: Katerina Krejci
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1    # sub-0001/ses-01

# get starting time:
start=`date +%s`
# FUNCTIONS
# ==============================================================================
# Segment rootlets if it does not exist
segment_rootlets_if_does_not_exist(){
  FILESEGROOTLETS="${1}_label-rootlets_dseg"
  FILESEGROOTLETSMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEGROOTLETS}.nii.gz"
  echo
  echo "Looking for manual rootlets segmentation: $FILESEGROOTLETSMANUAL"
  if [[ -e $FILESEGROOTLETSMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGROOTLETSMANUAL ${FILESEGROOTLETS}.nii.gz
    sct_qc -i ${1}.nii.gz -s ${1}_label-SC_mask.nii.gz -d ${1}_label-rootlets_dseg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Run model-spinal-rootlets_ventral_D106_r20240523 for dorsal and ventral rootlets segmentation
    # NOTE: as the model for both ventral and dorsal rootlets is not part of SCT yet, we run directly the nnUNet model using
    # the wrapper script run_inference_single_subject.py from the model-spinal-rootlets repository
    # https://github.com/ivadomed/model-spinal-rootlets/blob/main/packaging_ventral_rootlets/run_inference_single_subject.py
    # NOTE: we use SCT python because it has nnUNet installed
    # NOTE: the command below expects that you downloaded the model (https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240523) and saved it to:  ~/models/model-spinal-rootlets_ventral_D106_r20240523
    $SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/packaging_ventral_rootlets/run_inference_single_subject.py -i ${1}.nii.gz -o ${1}_label-rootlets_dseg.nii.gz -path-model ~/models/model-spinal-rootlets_ventral_D106_r20240523/model-spinal-rootlets_ventral_D106_r20240523 -fold all

    # Run sct_qc for quality control of rootlet segmentation.
    sct_qc -i ${1}.nii.gz -s ${1}_label-SC_mask.nii.gz -d ${1}_label-rootlets_dseg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
  fi
}


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we copy only T2w to save space
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*T2w.* .

# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat

# Define the names of files
file_inv_1=${SUBJECT}_acq-ND_inv-1_MP2RAGE
file_inv_2=${SUBJECT}_acq-ND_inv-2_MP2RAGE
file_unit_1=${SUBJECT}_acq-ND_UNIT1

# Segment rootlets for each file
segment_rootlets_if_does_not_exist ${file_inv_1}.nii.gz
segment_rootlets_if_does_not_exist ${file_inv_2}.nii.gz
segment_rootlets_if_does_not_exist ${file_unit_1}.nii.gz

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"