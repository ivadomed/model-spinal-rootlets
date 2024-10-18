#!/bin/bash
#
# Process data of different neck positions (extension, flexion and straight).
#
# Usage:
#   ./process_data.sh <SUBJECT>
# 
# Manual segmentations and labels (discs, PMJ, nerve rootlets) should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
#
# Authors: Sandrine Bédard

set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1

# Save script path
PATH_SCRIPT=$PWD

# get starting time:
start=`date +%s`

# FUNCTIONS
# ==============================================================================

# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t2
  fi
}

# Check if manual segmentation already exists. If it does, copy it locally. If it does not, perform segmentation.
segment_if_does_not_exist(){
  local file="$1"
  folder_contrast="anat"

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg -i ${file}.nii.gz -task seg_sc_contrast_agnostic -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if manual rootlets already exists. If it does, copy it locally. If it does not, perform segmentation.
segment_rootlets_if_does_not_exist(){
  local file="$1"
  # Update global variable with segmentation file name
  FILELABEL="${file}_rootlets"
  set +e
  FILELABELMANUAL="$(ls ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/*rootlets*.nii.gz)"
  set -e -o pipefail
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual rootlets labels."
    rsync -avzh $FILELABELMANUAL ${file}_rootlets.nii.gz
  else
    echo "Not found. Proceeding with automatic labeling."
    # Segment rootlets
    sct_deepseg -i ${file}.nii.gz -task seg_spinal_rootlets_t2w -qc ${PATH_QC} -o ${FILELABEL}.nii.gz

  fi
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

SUBJECT_ID=$(dirname "$SUBJECT")
SES=$(basename "$SUBJECT")
# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/
file_t2="${SUBJECT_ID}_${SES}_T2w"

# 1. Segment spinal cord
segment_if_does_not_exist $file_t2

# 2. Create C2-C3 disc label in the cord
label_if_does_not_exist $file_t2 ${file_t2}_seg
sct_label_utils -i ${file_t2}_seg_labeled_discs.nii.gz -keep 3 -o ${file_t2}_seg_labeled_discs3.nii.gz -qc ${PATH_QC} -qc-subject ${file_t2}
# Create mid-vertebrae labels
sct_label_utils -i ${file_t2}_seg_labeled.nii.gz -vert-body 2,7 -o ${file_t2}_seg_labeled_vertbody_27.nii.gz -qc ${PATH_QC} -qc-subject ${file_t2}

# 3. Segment rootlets
segment_rootlets_if_does_not_exist $file_t2
file_t2_rootlets=$FILELABEL
# Create center-of-mass of rootlets
sct_label_utils -i ${file_t2_rootlets}.nii.gz -cubic-to-point -o ${file_t2_rootlets}_mid.nii.gz
sct_label_utils -i ${file_t2}_seg.nii.gz -project-centerline ${file_t2_rootlets}_mid.nii.gz  -o ${file_t2_rootlets}_mid_center.nii.gz
sct_qc -i ${file_t2}.nii.gz  -s ${file_t2_rootlets}_mid_center.nii.gz -p sct_label_utils -qc $PATH_QC

# 4. Register T2-w image to PAM50 template # TODO: add time for each
# With rootlets
start_rootlets=`date +%s`
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -ldisc ${file_t2}_seg_labeled_discs3.nii.gz  -lrootlets ${file_t2_rootlets}.nii.gz  -ofolder reg_rootlets -qc $PATH_QC
#sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -ldisc ${file_t2_rootlets}_mid_center.nii.gz  -lrootlets ${file_t2_rootlets}.nii.gz  -ofolder reg_rootlets -qc $PATH_QC
end_rootlets=`date +%s`
runtime=$((end_rootlets-start_rootlets))
echo "+++++++++++ TIME: Duration of of rootlet reg2template:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
# bring spinal nerve rootlets segmentation in template space for comparison!
sct_apply_transfo -i ${file_t2_rootlets}.nii.gz -x nn -w reg_rootlets/warp_anat2template.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -o reg_rootlets/${file_t2_rootlets}_2template.nii.gz

# With all discs labels
start_discs=`date +%s`
sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -ldisc ${file_t2}_seg_labeled_discs.nii.gz -ofolder reg_discs -qc $PATH_QC
end_discs=`date +%s`
runtime=$((end_discs-start_discs))
echo "+++++++++++ TIME: Duration of of discs reg2template:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
# bring spinal nerve rootlets segmentation in template space for comparison!
sct_apply_transfo -i ${file_t2_rootlets}.nii.gz -x nn -w reg_discs/warp_anat2template.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -o reg_discs/${file_t2_rootlets}_2template.nii.gz



# # With 2 mid-vertebrae labels
# start_vert=`date +%s`
# sct_register_to_template -i ${file_t2}.nii.gz -s ${file_t2}_seg.nii.gz -l ${file_t2}_seg_labeled_vertbody_27.nii.gz -ofolder reg_midvert_c2c7 -qc $PATH_QC
# end_vert=`date +%s`
# runtime=$((end_vert-start_vert))
# echo "+++++++++++ TIME: Duration of of vert reg2template:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"

# TODO: save csv file with time

# Display useful info for the log
# ===============================
end=`date +%s`
runtime_final=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime_final / 3600))hrs $((($runtime_final / 60) % 60))min $(($runtime_final % 60))sec"
echo "~~~"
