#!/bin/bash
#
# This script performs:
# - segmentation of spinal cord from T2w data (seg_sc_contrast_agnostic)
# - detection of PMJ from T2w data (sct_detect_pmj).
# - generation of intervertebral disc labels from T2w data (sct_label_vertebrae)
# - projection of intervertebral disc labels to the spinal cord centerline (sct_label_utils)
# - finding the rootlets segmentation (if it exists)
# - computing the spinal levels of the rootlets (if the rootlets segmentation exists) and distances to the PMJ
#
# Expected file naming (cropped data):
#   image:        sub-XXX_<contrast>_crop.nii.gz                 (e.g. sub-amuAL_T1w_crop.nii.gz)
#   SC seg:       sub-XXX_<contrast>_label-SC_seg_crop.nii.gz    (e.g. sub-amuAL_T1w_label-SC_seg_crop.nii.gz)
#   PMJ:          sub-XXX_<contrast>_label-pmj_crop.nii.gz
#   discs:        sub-XXX_<contrast>_label-disc-manual_crop.nii.gz
#   rootlets:     sub-XXX_<contrast>_label-rootletseg_crop.nii.gz

# NOTE: This script is inspired by the script 'inter-rater_variability/02_run_batch_inter_rater_variability.sh'
# https://github.com/ivadomed/model-spinal-rootlets/blob/main/inter-rater_variability/02_run_batch_inter_rater_variability.sh

# This script used the script '02a_rootlets_to_spinal_levels.py' (to get spinal levels) available at:
# https://github.com/ivadomed/model-spinal-rootlets/blob/main/inter-rater_variability/02a_rootlets_to_spinal_levels.py

# Usage:
## sct_run_batch -script analysis_preprocess_pipeline.sh
##                     -path-data <DATA>
##                     -path-output <DATA>_202X-XX-XX
##                     -jobs 5

# Authors: Katerina Krejci


# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=${1%%/*}

# Get starting time:
start=`date +%s`


# FUNCTIONS
# ==============================================================================
# Uses global variables:
#   base     = sub-XXX_<contrast>          (e.g. sub-amuAL_T1w)
#   file     = sub-XXX_<contrast>_crop     (cropped image, without .nii.gz)
#   contrast = T1w | T2w

# Segment spinal cord if it does not exist in the derivatives folder
segment_sc_if_does_not_exist(){
  FILESEG="${base}_label-SC_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_image -i ${FILESEG}.nii.gz -setorient RPI -o ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1  sct_deepseg -task seg_sc_contrast_agnostic -i ${file}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}

# Detect PMJ if it does not exist in the derivatives folder
detect_pmj_if_does_not_exist(){
  FILEPMJ="${base}_label-pmj"
  FILEPMJMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILEPMJ}.nii.gz"
  echo
  echo "Looking for manual PMJ detection: $FILEPMJMANUAL"
  if [[ -e $FILEPMJMANUAL ]]; then
    echo "Found! Using manual PMJ detection."
    rsync -avzh $FILEPMJMANUAL ${FILEPMJ}.nii.gz
    sct_image -i ${FILEPMJ}.nii.gz -setorient RPI -o ${FILEPMJ}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILEPMJ}.nii.gz -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic PMJ detection."

    # if the contrast is T2w, use t2 for PMJ detection; otherwise use t1
    if [[ $contrast == "T2w" ]]; then
      sct_detect_pmj -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t2 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      sct_detect_pmj -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t1 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
  fi
}

# Label vertebral levels if it does not exist in the derivatives folder
label_if_does_not_exist(){
  FILELABEL="${base}_label-disc-manual"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual intervertebral disc labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    sct_image -i ${FILELABEL}.nii.gz -setorient RPI -o ${FILELABEL}.nii.gz
  else
    echo "Manual intervertebral discs not found. Proceeding with automatic labeling."

    if [[ $contrast == "T2w" ]]; then
      sct_label_vertebrae -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      sct_label_vertebrae -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t1 -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
    # Rename automatically generated disc labels to match the manual ones
    mv ${FILESEG}_labeled_discs.nii.gz ${FILELABEL}.nii.gz
  fi
  # Generate QC report for intervertebral disc labeling (either manual or automatic)
  sct_qc -i ${file}.nii.gz -s ${FILELABEL}.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# Copy rootlets segmentation if it exists in the derivatives folder
copy_rootlets_if_exist(){
  FILESEGROOTLETS="${base}_label-rootlets_dseg"
  FILESEGROOTLETSMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEGROOTLETS}.nii.gz"
  echo
  echo "Looking for manual rootlets segmentation: $FILESEGROOTLETSMANUAL"
  if [[ -e $FILESEGROOTLETSMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGROOTLETSMANUAL ${FILESEGROOTLETS}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -d ${FILESEGROOTLETS}.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
  else
    echo "Not found. Creating automatic rootlets segmentation."
    CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 sct_deepseg rootlets -i ${file}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEGROOTLETS}.nii.gz
  fi
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*.* .

# Go to anat folder where all data are located
cd ${SUBJECT}/anat

echo "SUBJECT=${SUBJECT}"
echo "PWD=$(pwd)"
ls

for contrast in T1w T2w; do
    base="${SUBJECT//[\/]/_}_${contrast}"      # e.g. sub-amuAL_T1w
    file="${base}"                        # e.g. sub-amuAL_T1w_crop

    if [[ ! -f ${file}.nii.gz ]]; then
        echo "WARNING: ${file}.nii.gz not found in $(pwd), skipping."
        continue
    fi

    # Segment spinal cord (only if it does not exist)
    segment_sc_if_does_not_exist

    # Run sct_label_vertebrae for vertebral levels estimation (not needed for now)
    #label_if_does_not_exist

    # Project the intervertebral disc labels to the spinal cord centerline (not needed for now)
    #sct_label_utils -i ${FILESEG}.nii.gz -o ${FILELABEL}_centerline.nii.gz -project-centerline ${FILELABEL}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

    # Detect PMJ (only if it does not exist)
    detect_pmj_if_does_not_exist

    # Copy the rootlets segmentation if it exists
    copy_rootlets_if_exist

    # Get rootlets spinal levels
    # Note: we use SCT python because the script imports some SCT classes
    $SCT_DIR/python/envs/venv_sct/bin/python /code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py \
        -i ${FILESEGROOTLETS}.nii.gz -s ${FILESEG}.nii.gz -pmj ${FILEPMJ}.nii.gz -dilate 3

    # Copy the CSV file with the spinal levels distances from the PMJ to the results folder (used by
    # 02_compute_cervical_midpoints_distance.py), so that only the results folder needs to be copied from the server
    rsync -avzh ${FILESEGROOTLETS}_pmj_distance.csv ${PATH_RESULTS}/

done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
