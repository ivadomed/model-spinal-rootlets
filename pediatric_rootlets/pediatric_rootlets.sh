#!/bin/bash
#
# This script performs:
# - segmentation of rootlets from T2w data (model model-spinal-rootlets_ventral_D106_r20240523),
# - segmentation of spinal cord from T2w data (contrast agnostic model)
# - segmentation of vertebral levels from T2w data (sct_label_vertebrae)
# - detection of PMJ from T2w data (sct_detect_pmj).
#
# Authors: Katerina Krejci, Jan Valosek
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

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
cp -r $PATH_DATA/${SUBJECT%/*} .    # %/* - delete everything after last slash (/)

# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat

# Run model-spinal-rootlets_ventral_D106_r20240523 for dorsal and ventral rootlets segmentation
# NOTE: as the model for both ventral and dorsal rootlets is not part of SCT yet, we run directly the nnUNet model using
# the wrapper script run_inference_single_subject.py from the model-spinal-rootlets repository
# https://github.com/ivadomed/model-spinal-rootlets/blob/main/packaging_ventral_rootlets/run_inference_single_subject.py
# NOTE: we use SCT python because it has nnUNet installed 
# NOTE: the command below expects that you downloaded the model (https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240523) and saved it to:  ~/models/model-spinal-rootlets_ventral_D106_r20240523

file_t2=${SUBJECT}_rec-composed_T2w.nii.gz

$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/packaging_ventral_rootlets/run_inference_single_subject.py -i ${file_t2} -o ${SUBJECT}_T2w_label-rootlets_dseg.nii.gz -path-model ~/models/model-spinal-rootlets_ventral_D106_r20240523/model-spinal-rootlets_ventral_D106_r20240523 -fold all

# Segmentation of spinal cord from T2w data
# NOTE: on two testing subjects, the contrast agnostic model performed better than sct_deepseg_sc -- it might be
# interesting to compare the two methods
sct_deepseg -task seg_sc_contrast_agnostic -i ${file_t2} -o ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Run sct_qc for quality control of rootlet segmentation. We are running the QC here because we need also the SC seg to
# crop the image
sct_qc -i ${file_t2} -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -d ${SUBJECT}_T2w_label-rootlets_dseg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial

# Run sct_label_vertebrae for vertebral levels estimation
sct_label_vertebrae -i ${file_t2} -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Run sct_detect_pmj for PMJ detection
sct_detect_pmj -i ${file_t2} -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -c t2 -o ${SUBJECT}_acq-top_run-1_T2w_pmj_ca.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Get rootlets spinal levels
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${SUBJECT}_T2w_label-rootlets_dseg.nii.gz -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -pmj ${SUBJECT}_acq-top_run-1_T2w_pmj_ca.nii.gz -type rootlets

# Get vertebral spinal levels - with cropping parts, where are more than just 2 levels (background and level)
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca_labeled.nii.gz -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -pmj ${SUBJECT}_acq-top_run-1_T2w_pmj_ca.nii.gz -type vertebral



# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"