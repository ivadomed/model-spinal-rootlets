#!/bin/bash
#
# This script performs:
# - segmentation of spinal cord from T2w data (seg_sc_contrast_agnostic)
# - detection of PMJ from T2w data (sct_detect_pmj).
# - generation of intervertebral disc labels from T2w data (sct_label_vertebrae)
# - projection of intervertebral disc labels to the spinal cord centerline (sct_label_utils)
# - finding the rootlets segmentation (if it exists)

# NOTE: Rootlets segmentation is not performed in this script. We used the nnUNet predictions of MULTICON model instead
# of SCT. When the segmentation model will be included in SCT, we will use it instead of the nnUNet.

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
SUBJECT=$1    # sub-0001/ses-01

# Get starting time:
start=`date +%s`


# FUNCTIONS
# ==============================================================================
# Segment spinal cord if it does not exist in the derivatives folder
segment_sc_if_does_not_exist(){
  FILESEG="${file}_SC_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/${SUBJECT}/anat/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    sct_image -i $FILESEGMANUAL -setorient RPI -o $FILESEGMANUAL
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    sct_deepseg -task seg_sc_contrast_agnostic -i ${file}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}

# Detect PMJ if it does not exist in the derivatives folder
detect_pmj_if_does_not_exist(){
  FILEPMJ="${file}_labels-pmj-manual"
  FILEPMJMANUAL="${PATH_DATA}/derivatives/${SUBJECT}/anat/${FILEPMJ}.nii.gz"
  echo
  echo "Looking for manual PMJ detection: $FILEPMJMANUAL"
  if [[ -e $FILEPMJMANUAL ]]; then
    echo "Found! Using manual PMJ detection."
    sct_image -i $FILEPMJMANUAL -setorient RPI -o $FILEPMJMANUAL
    rsync -avzh $FILEPMJMANUAL ${FILEPMJ}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILEPMJ}.nii.gz -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic PMJ detection."

    # if the contrast is T2w, use T2w contrast (OpenNeuro and SpineGeneric data) for PMJ detection; otherwise,
    # use T1w contrast (MP2RAGE data)
    if [[ $file == *"_ses-headNormal_T2w" ]] || [[ $file == *"_T2w" ]]; then
      sct_detect_pmj -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t2 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      sct_detect_pmj -i ${file}.nii.gz -s ${FILESEG}.nii.gz -c t1 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
  fi
}

# Label vertebral levels if it does not exist in the derivatives folder
label_if_does_not_exist(){
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual intervertebral disc labels."
    sct_image -i $FILELABELMANUAL -setorient RPI -o $FILELABELMANUAL
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
  else
    echo "Manual intervertebral discs not found. Proceeding with automatic labeling."

    # if the contrast is T2w, use T2w contrast (OpenNeuro and SpineGeneric data) for vertebral labeling; otherwise,
    # use T1w contrast (MP2RAGE data)
    if [[ $file == *"_ses-headNormal_T2w" ]] || [[ $file == *"_T2w" ]]; then
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
  FILESEGROOTLETS="${file}_rootletsseg"
  FILESEGROOTLETSMANUAL="${PATH_DATA}/derivatives/${SUBJECT}/anat/${FILESEGROOTLETS}.nii.gz"
  echo
  echo "Looking for manual rootlets segmentation: $FILESEGROOTLETSMANUAL"
  if [[ -e $FILESEGROOTLETSMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGROOTLETSMANUAL ${FILESEGROOTLETS}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${file}_SC_seg.nii.gz -d ${file}_rootletsseg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
  else
    echo "Not found."
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

# Define the names of the T2w files and the MP2RAGE file (INV2 contrast)
file_t2w_open_neuro=${SUBJECT}_ses-headNormal_T2w
file_mp2rage=${SUBJECT}_inv-2_part-mag_MP2RAGE
file_t2w_spine_generic=${SUBJECT}_T2w

# Check if the T2w file exists
if [ -f ${file_t2w_open_neuro}.nii.gz ]; then
    file=${file_t2w_open_neuro}
    echo "Processing file from Open Neuro dataset."

# Check if the MP2RAGE file exists
elif [ -f ${file_mp2rage}.nii.gz ]; then
  file=${file_mp2rage}
  echo "Processing file from MP2RAGE dataset."

# Check if the T2w file from Spine Generic dataset exists
elif [ -f ${file_t2w_spine_generic}.nii.gz ]; then
    file=${file_t2w_spine_generic}
    echo "Processing file from Spine Generic dataset."
fi

 # Segment spinal cord (only if it does not exist)
segment_sc_if_does_not_exist ${file}.nii.gz

# Run sct_label_vertebrae for vertebral levels estimation
label_if_does_not_exist ${file}.nii.gz

# Project the intervertebral disc labels to the spinal cord centerline
sct_label_utils -i ${FILESEG}.nii.gz -o ${FILELABEL}_centerline.nii.gz -project-centerline ${FILELABEL}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Detect PMJ (only if it does not exist)
detect_pmj_if_does_not_exist ${file}.nii.gz

# Copy the rootlets segmentation if it exists
copy_rootlets_if_exist ${file}.nii.gz

# Get rootlets spinal levels
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${file}_rootletsseg.nii.gz -s ${file}_SC_seg.nii.gz -pmj ${file}_labels-pmj-manual.nii.gz -dilate 3

# Get vertebral levels - according to the intervertebral discs
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/discs_to_vertebral_levels.py -centerline ${file}_SC_seg_centerline_extrapolated.csv -disclabel ${file}_labels-disc_centerline.nii.gz


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
