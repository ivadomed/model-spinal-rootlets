#!/usr/bin/env bash
#script to open MRI and mask, arg 1: subject number, arg 2: headposition
sub_num="$1"
pos_type="$2"

command="fsleyes sub-${1}/ses-head${2}/anat/sub-${1}_ses-head${2}_T2w.nii.gz -cm greyscale  derivatives/labels/sub-${1}/ses-head${2}/anat/sub-${1}_ses-head${2}_T2w_root-manual.nii.gz -cm red-yellow"

eval $command
