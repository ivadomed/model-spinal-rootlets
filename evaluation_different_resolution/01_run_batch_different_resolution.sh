#!/bin/bash
#
# The script runs our nnUNet model on a single subject (sub-010_ses-headUp) from the Spinal Cord Head Position MRI
# dataset (https://openneuro.org/datasets/ds004507/versions/1.0.1). 0.6mm iso T2w image of this subject was resampled
# to 0.8mm, 1.0mm, 1.2mm, 1.4mm, and 1.5mm iso resolution using the sct_resample function to evaluate the performance
# of our nnUNet model on different resolutions.
#
# Namely:
#   - segment spinal cord (using SCT)
#   - detect PMJ label (using SCT)
#   - segment rootlets (using our nnUNet model)
#   - run inter-rater_variability/02a_rootlets_to_spinal_levels.py to project the nerve rootlets on the spinal cord
#   segmentation to obtain spinal levels, and compute the distance between the pontomedullary junction (PMJ) and the
#   start and end of the spinal level. The inter-rater_variability/02a_rootlets_to_spinal_levels.py script outputs
#   .nii.gz file with spinal levels and saves the results in CSV files.
#
# The script requires the SCT conda environment to be activated:
#    source ${SCT_DIR}/python/etc/profile.d/conda.sh
#    conda activate venv_sct
#
# TODO: the script also needs nnunet conda environment to be activated, currently path to the nnunet python is
#  hard-coded --> explore if two venvs can be activated at the same time
#
# Usage:
#       sct_run_batch -script 01_run_batch_different_resolution.sh -path-data <DATA> -path-output <DATA>_202X-XX-XX -jobs 5 -script-args "<PATH_REPO> <PATH_TO_NNUNET_MODEL> <FOLD>"
#
# Authors: Jan Valosek
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# Retrieve input params and other params
SUBJECT=$1
# Path to this repository
PATH_REPO=$2
PATH_NNUNET_MODEL=$3
FOLD=$4

echo "SUBJECT: ${SUBJECT}"
echo "PATH_REPO: ${PATH_REPO}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"
echo "FOLD: ${FOLD}"

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Check if manual spinal cord segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Segment rootlets using our nnUNet model
segment_rootlets_nnUNet(){
  local file="$1"

  echo "Segmenting rootlets using our nnUNet model."
  # Run rootlets segmentation
  # TODO: the hard-coded path to the conda environment is not ideal. But the script also needs to be run inside the
  #  sct_venv environment --> explore if two venvs can be activated at the same time
  ${HOME}/miniconda3/envs/nnunet/bin/python ${PATH_REPO}/packaging/run_inference_single_subject.py -i ${file}.nii.gz -o ${file}_label-rootlet_nnunet.nii.gz -path-model ${PATH_NNUNET_MODEL} -fold ${FOLD}
}

# Check if manual PMJ label file already exists. If it does, copy it locally.
# If it doesn't, perform automatic PMJ labeling
detect_pmj() {
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_pmj"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual PMJ label: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual PMJ label."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_label_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_detect_pmj -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Copy GT segmentation
copy_gt(){
  local file="$1"
  local suffix="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_${suffix}.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEGMANUAL ${file}_${suffix}.nii.gz
  else
      echo "File ${FILESEGMANUAL} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILESEGMANUAL} does not exist. Exiting."
      exit 1
  fi
}

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECT .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECT//[\/]/_}"

# -------------------------------------------------------------------------
# T2w
# -------------------------------------------------------------------------
# Add suffix corresponding to contrast
file_t2w=${file}_T2w
# Check if T2w image exists
if [[ -f ${file_t2w}.nii.gz ]];then

    # Segment spinal cord
    segment_if_does_not_exist $file_t2w t2

    # Label PMJ
    detect_pmj $file_t2w t2

    # Segment rootlets using our nnUNet model
    segment_rootlets_nnUNet $file_t2w

    # Project the nerve rootlets on the spinal cord segmentation to obtain spinal levels and compute the distance
    # between the pontomedullary junction (PMJ) and the start and end of the spinal level
    python ${PATH_REPO}/inter-rater_variability/02a_rootlets_to_spinal_levels.py -i ${file_t2w}_label-rootlet_nnunet.nii.gz -s ${file_t2w}_seg.nii.gz -pmj ${file_t2w}_pmj.nii.gz

fi
# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"