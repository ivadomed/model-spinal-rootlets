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
segment_if_does_not_exist(){
  FILESEG="${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}
    sct_qc -i ${file_t2} -s ${FILESEG} -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file_t2} -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}
  fi
}

# Detect PMJ if it does not exist
detect_pmj_if_does_not_exist(){
  FILEPMJ="${SUBJECT}_${FILE_TYPE}_T2w_pmj.nii.gz"
  FILEPMJMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILEPMJ}"
  echo
  echo "Looking for manual PMJ detection: $FILEPMJMANUAL"
  if [[ -e $FILEPMJMANUAL ]]; then
    echo "Found! Using manual PMJ detection."
    rsync -avzh $FILEPMJMANUAL ${FILEPMJ}
    sct_qc -i ${file_t2} -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -d ${FILEPMJ} -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic PMJ detection."
    # Detect PMJ
    sct_detect_pmj -i ${file_t2} -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -c t2 -o ${FILEPMJ} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Label vertebral levels if it does not exist
label_if_does_not_exist(){
  # Update global variable with segmentation file name
  FILELABEL="${file_t2}_labels-disc.nii.gz"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file_t2} -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -discfile ${FILELABEL} -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file_t2} -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

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

# Define the names of the T2w files
file_t2_composed=${SUBJECT}_rec-composed_T2w.nii.gz
file_t2_top=${SUBJECT}_acq-top_run-1_T2w.nii.gz

# Check if file_t2_composed exists, if not, use file_t2_top as file_t2
if [ -f ${file_t2_composed} ]; then
    file_t2=${file_t2_composed}
    FILE_TYPE="rec-composed"
else
    file_t2=${file_t2_top}
    FILE_TYPE="top"
    echo "Composed T2w file not found. Proceeding only with Top T2w file."
fi


# Run model-spinal-rootlets_ventral_D106_r20240523 for dorsal and ventral rootlets segmentation
# NOTE: as the model for both ventral and dorsal rootlets is not part of SCT yet, we run directly the nnUNet model using
# the wrapper script run_inference_single_subject.py from the model-spinal-rootlets repository
# https://github.com/ivadomed/model-spinal-rootlets/blob/main/packaging_ventral_rootlets/run_inference_single_subject.py
# NOTE: we use SCT python because it has nnUNet installed
# NOTE: the command below expects that you downloaded the model (https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240523) and saved it to:  ~/models/model-spinal-rootlets_ventral_D106_r20240523
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/packaging_ventral_rootlets/run_inference_single_subject.py -i ${file_t2} -o ${SUBJECT}_T2w_${FILE_TYPE}_label-rootlets_dseg.nii.gz -path-model ~/models/model-spinal-rootlets_ventral_D106_r20240523/model-spinal-rootlets_ventral_D106_r20240523 -fold all

 # Segment spinal cord (only if it does not exist)
segment_if_does_not_exist ${file_t2}

# Run sct_qc for quality control of rootlet segmentation. We are running the QC here because we need also the SC seg to
# crop the image
sct_qc -i ${file_t2} -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -d ${SUBJECT}_T2w_${FILE_TYPE}_label-rootlets_dseg.nii.gz -p sct_deepseg_lesion -qc ${PATH_QC} -qc-subject ${SUBJECT} -plane axial

# Run sct_label_vertebrae for vertebral levels estimation
label_if_does_not_exist ${file_t2}

# Detect PMJ (only if it does not exist)
detect_pmj_if_does_not_exist ${file_t2}

# Get rootlets spinal levels
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${SUBJECT}_T2w_${FILE_TYPE}_label-rootlets_dseg.nii.gz -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -pmj ${SUBJECT}_${FILE_TYPE}_T2w_pmj.nii.gz -type rootlets

# Get vertebral spinal levels - with cropping parts, where are more than just 2 levels (background and level)
# Note: we use SCT python because the `02a_rootlets_to_spinal_levels.py` script imports some SCT classes
$SCT_DIR/python/envs/venv_sct/bin/python ~/code/model-spinal-rootlets/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask_labeled.nii.gz -s ${SUBJECT}_${FILE_TYPE}_T2w_label-SC_mask.nii.gz -pmj ${SUBJECT}_${FILE_TYPE}_T2w_pmj.nii.gz -type vertebral


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"