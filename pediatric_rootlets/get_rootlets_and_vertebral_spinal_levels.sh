#!/bin/bash
#
# This script performs:
# - run 02a_rootlets_to_spinal_levels.py for rootlets to spinal levels on pediatric data
# - run 02a_rootlets_to_spinal_levels.py for vertebrae to spinal levels on pediatric data (with selecting only
# 2 levels in each slice - background (0) and one level -> if there are more than 2 levels in one slice,
# the script will not concider this slice)
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

# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat

# for rootlets spinal levels
python 02a_rootlets_to_spinal_levels.py -i ${SUBJECT}_T2w_label-rootlets_dseg.nii.gz -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -pmj ${SUBJECT}_acq-top_run-1_T2w_pmj_ca.nii.gz -type rootlets

# for vertebral spinal levels - with cropping parts, where are more than just 2 levels (background and level)
python 02a_rootlets_to_spinal_levels.py -i ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca_labeled.nii.gz -s ${SUBJECT}_acq-top_run-1_T2w_deepseg_ca.nii.gz -pmj ${SUBJECT}_acq-top_run-1_T2w_pmj_ca.nii.gz -type vertebral


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"