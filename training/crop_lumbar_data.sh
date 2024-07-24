#!/bin/bash

# Crop lumbar data around the spinal cord
# The script iterates over all the images in the current directory (`imagesTr`) and:
#   1. Segments the spinal cord using the contrast-agnostic model
#   2. Crops the image and label around the spinal cord
#
# Example usage:
#     cd ~/data/Dataset301_LumbarRootlets_crop/imagesTr
#     bash ~/code/model-spinal-rootlets/training/crop_lumbar_data.sh
#
#
# The folder structure is expected to be:
#   ├── dataset.json
#   ├── imagesTr
#   │    ├── sub-CTS04_ses-SPpre_T2w_001_0000.nii.gz
#   │    ├── sub-CTS05_ses-SPpre_T2w_001_0000.nii.gz
#   │    ├── sub-CTS09_ses-SPpre_T2w_001_0000.nii.gz
#   │    ├── sub-CTS10_ses-SPanat_T2w_001_0000.nii.gz
#   │    ├── sub-CTS14_ses-SPpre_T2w_001_0000.nii.gz
#   │    └── sub-CTS15_ses-SPpre_T2w_001_0000.nii.gz
#   └── labelsTr
#       ├── sub-CTS04_ses-SPpre_T2w_001.nii.gz
#       ├── sub-CTS05_ses-SPpre_T2w_001.nii.gz
#       ├── sub-CTS09_ses-SPpre_T2w_001.nii.gz
#       ├── sub-CTS10_ses-SPanat_T2w_001.nii.gz
#       ├── sub-CTS14_ses-SPpre_T2w_001.nii.gz
#       └── sub-CTS15_ses-SPpre_T2w_001.nii.gz
#
# Author: Jan Valosek
#

for file in *.nii.gz; do
    echo "Processing $file"
    file=${file/.nii.gz/}
    file_label=${file/_0000/}
    # Segment the spinal cord
    sct_deepseg -i ${file}.nii.gz -o ${file}_seg.nii.gz -task seg_sc_contrast_agnostic -qc ../qc -qc-subject ${file}
    # Crop the image around the spinal cord
    sct_crop_image -i ${file}.nii.gz -m ${file}_seg.nii.gz -dilate 64x64x64 -o ${file}.nii.gz
    # Crop the label around the spinal cord located in labelsTr
    sct_crop_image -i ../labelsTr/${file_label}.nii.gz -m ${file}_seg.nii.gz -dilate 64x64x64 -o ../labelsTr/${file_label}.nii.gz
    # Remove the segmentation mask
    rm ${file}_seg.nii.gz
done