#!/bin/bash
#
# The script runs our nnUNet model on the subject sub-01 with 10 sessions from the open-access courtois-neuromod dataset
# to evaluate the inter-subject stability.
#
# Namely:
#   - segment spinal cord (using SCT)
#   - detect PMJ label (using SCT)
#   - segment rootlets (using our nnUNet model)
#   - run 02a_rootlets_to_spinal_levels.py to project the nerve rootlets on the spinal cord segmentation to obtain spinal
#   levels, and compute the distance between the pontomedullary junction (PMJ) and the start and end of the spinal
#   level. The inter-rater_variability/02a_rootlets_to_spinal_levels.py script outputs .nii.gz file with spinal levels
#   and saves the results in CSV files.
#
# The script requires the SCT conda environment to be activated:
#    source ${SCT_DIR}/python/etc/profile.d/conda.sh
#    conda activate venv_sct
#
# Usage:
#       sct_run_batch -script 02a_courtois-neuromod-run_prediction.sh
#                     -path-data <DATA>
#                     -path-output <DATA>_202X-XX-XX
#                     -jobs 1
#                     -include sub-01
#                     -script-args "<PATH_TO_PYTHON_SCRIPTS> <PATH_TO_NNUNET_SCRIPT> <PATH_TO_NNUNET_MODEL>"
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
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
# Path to the directory with 02a_rootlets_to_spinal_levels.py script
PATH_PYTHON_SCRIPTS=$2
PATH_NNUNET_SCRIPT=$3
PATH_NNUNET_MODEL=$4

echo "SUBJECT: ${SUBJECT}"
echo "PATH_PYTHON_SCRIPTS: ${PATH_PYTHON_SCRIPTS}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

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
  ${HOME}/miniconda3/envs/nnunet/bin/python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${file}_label-rootlet_nnunet.nii.gz -path-model ${PATH_NNUNET_MODEL} -use-best-checkpoint
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
file_t2w="${file}_bp-cspine_T2w"
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
    python ${PATH_PYTHON_SCRIPTS}/02a_rootlets_to_spinal_levels.py -i ${file_t2w}_label-rootlet_nnunet.nii.gz -s ${file_t2w}_seg.nii.gz -pmj ${file_t2w}_pmj.nii.gz

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