#!/bin/bash
#
# This script performs:
# - segmentation of rootlets from T2w data (model model-spinal-rootlets_ventral_D106_r20240523),
# - segmentation of spinal cord from T2w data (sct_deepseg_sc)
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
# FUNCTIONS
# ==============================================================================

# Segment spinal cord if it does not exist
segment_sc_if_does_not_exist(){
  FILESEG="${file_t2}_label-SC_mask"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file_t2}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}

# Detect PMJ if it does not exist
detect_pmj_if_does_not_exist(){
  FILEPMJ="${file_t2}_label-PMJ_dlabel"
  FILEPMJMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILEPMJ}.nii.gz"
  echo
  echo "Looking for manual PMJ detection: $FILEPMJMANUAL"
  if [[ -e $FILEPMJMANUAL ]]; then
    echo "Found! Using manual PMJ detection."
    rsync -avzh $FILEPMJMANUAL ${FILEPMJ}.nii.gz
    sct_qc -i ${file_t2}.nii.gz -s ${FILEPMJ}.nii.gz -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic PMJ detection."
    # Detect PMJ
    sct_detect_pmj -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -c t2 -o ${FILEPMJ}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Label vertebral levels if it does not exist
label_if_does_not_exist(){
  # Update global variable with segmentation file name
  FILELABEL="${file_t2}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual intervertebral disc labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
  else
    echo "Manual intervertebral discs not found. Proceeding with automatic labeling."
    # Generate labeled segmentation and disc labels
    sct_label_vertebrae -i ${file_t2}.nii.gz -s ${FILESEG}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
    # Rename automatically generated disc labels to match the manual ones (otherwise, we would need conditionsto
    # handle different filenames)
    mv ${FILESEG}_labeled_discs.nii.gz ${FILELABEL}.nii.gz
  fi
  # Generate QC report for intervertebral disc labeling (either manual or automatic)
  sct_qc -i ${file_t2}.nii.gz -s ${FILELABEL}.nii.gz -p sct_label_utils -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# Segment rootlets if it does not exist
segment_rootlets_if_does_not_exist(){
  FILESEGROOTLETS="${file_t2}_label-rootlets_dseg"
  FILESEGROOTLETSMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEGROOTLETS}.nii.gz"
  echo
  echo "Looking for manual rootlets segmentation: $FILESEGROOTLETSMANUAL"
  if [[ -e $FILESEGROOTLETSMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGROOTLETSMANUAL ${FILESEGROOTLETS}.nii.gz
    MODIF=""
    sct_qc -i ${file_t2}.nii.gz -s ${file_t2}_label-SC_mask.nii.gz -d ${file_t2}_label-rootlets_dseg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Run model-spinal-rootlets_ventral_D106_r20240523 for dorsal and ventral rootlets segmentation
    # NOTE: as the model for both ventral and dorsal rootlets is not part of SCT yet, we run directly the nnUNet model using
    # the wrapper script run_inference_single_subject.py from the model-spinal-rootlets repository
    # https://github.com/ivadomed/model-spinal-rootlets/blob/main/packaging_ventral_rootlets/run_inference_single_subject.py
    # NOTE: we use SCT python because it has nnUNet installed
    # NOTE: the command below expects that you downloaded the model (https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240523) and saved it to:  ~/models/model-spinal-rootlets_ventral_D106_r20240523
    $SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/packaging_ventral_rootlets/run_inference_single_subject.py -i ${file_t2}.nii.gz -o ${file_t2}_label-rootlets_dseg.nii.gz -path-model ~/models/model-spinal-rootlets_ventral_D106_r20240523/model-spinal-rootlets_ventral_D106_r20240523 -fold all

    # Zeroing rootlets segmentation under the selected intervertebral disc
    $SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/pediatric_rootlets/zeroing_false_positive_rootlets.py -rootlets-seg ${file_t2}_label-rootlets_dseg.nii.gz -d ${FILELABEL}_centerline.nii.gz -x 11

    # Run sct_qc for quality control of rootlet segmentation.
    sct_qc -i ${file_t2}.nii.gz -s ${file_t2}_label-SC_mask.nii.gz -d ${file_t2}_label-rootlets_dseg_modif.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial
    MODIF="_modif"
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

# Define the names of the T2w files
file_t2_composed=${SUBJECT}_rec-composed_T2w
file_t2_top=${SUBJECT}_acq-top_run-1_T2w

# Check if file_t2_composed exists, if not, use file_t2_top as file_t2
if [ -f ${file_t2_composed}.nii.gz ]; then
    file_t2=${file_t2_composed}

# Use `acq-top` if the composed image does not exist. This is the case, for example, for `sub-125`.
else
    file_t2=${file_t2_top}
    echo "Composed T2w file not found. Proceeding only with Top T2w file."
fi

 # Segment spinal cord (only if it does not exist)
segment_sc_if_does_not_exist ${file_t2}.nii.gz

# Run sct_label_vertebrae for vertebral levels estimation
label_if_does_not_exist ${file_t2}.nii.gz

# Project the intervertebral disc labels to the spinal cord centerline
sct_label_utils -i ${FILESEG}.nii.gz -o ${FILELABEL}_centerline.nii.gz -project-centerline ${FILELABEL}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Detect PMJ (only if it does not exist)
detect_pmj_if_does_not_exist ${file_t2}.nii.gz

# Segment rootlets (only if it does not exist)
segment_rootlets_if_does_not_exist ${file_t2}.nii.gz

# Get rootlets spinal levels
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${file_t2}_label-rootlets_dseg${MODIF}.nii.gz -s ${file_t2}_label-SC_mask.nii.gz -pmj ${file_t2}_label-PMJ_dlabel.nii.gz

# Get vertebral levels - according to the intervertebral discs
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/pediatric_rootlets/discs_to_vertebral_levels.py -centerline ${file_t2}_label-SC_mask_centerline_extrapolated.csv -disclabel ${file_t2}_labels-disc_centerline.nii.gz


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"