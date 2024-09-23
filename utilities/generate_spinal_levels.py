#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# This script generates the spinal segments for the PAM50 template. It is based on the article by Frostell et al.
# https://www.frontiersin.org/articles/10.3389/fneur.2016.00238/full
# For more context, see: https://github.com/spinalcordtoolbox/PAM50/issues/16
# 
# How to run:
#   cd where this script is located and run:
#   python generate_spinal_levels.py
#   This will generate the file "PAM50_spinal_levels.nii.gz"
# 
# Author: Julien Cohen-Adad

import numpy as np
import nibabel as nib


# Identification of the slice on the PAM50 template that corresponds to the upper portion of the C1 nerve rootlets
z_top = 985
# Identification of the slice on the PAM50 template that corresponds to the caudal end of the spinal cord
z_bottom = 40

# Compute the length of the spinal cord (in mm), knowing that the pixel size along Z is 0.5mm.
length_spinalcord = z_top - z_bottom
length_spinalcord_mm = 0.5 * length_spinalcord

# Build dictionary of spinal segment location based on Table 3 of Frostell et al. article
percent_length_segment = [
    {"C1": 1.6},
    {"C2": 2.2},
    {"C3": 3.5},
    {"C4": 3.5},
    {"C5": 3.5},
    {"C6": 3.3},
    {"C7": 3.2},
    {"C8": 3.4},
    {"T1": 3.6},
    {"T2": 3.9},
    {"T3": 4.4},
    {"T4": 5},
    {"T5": 5.1},
    {"T6": 5.6},
    {"T7": 5.6},
    {"T8": 5.4},
    {"T9": 5.1},
    {"T10": 4.7},
    {"T11": 4.3},
    {"T12": 3.9},
    {"L1": 3.6},
    {"L2": 2.8},
    {"L3": 2.4},
    {"L4": 2.2},
    {"L5": 1.7},
    {"S1": 1.5},
    {"S2": 1.6},
    {"S3": 1.4},
    {"S4": 1.3},
    {"S5": 0.9},
]

print(f"Number of segments: {len(percent_length_segment)}")

# Verify that the sum of all relative length segment is 100
total_sum = 0.0
for item in percent_length_segment:
    value = list(item.values())[0]
    total_sum += value
print(f"Sum across percent segments: {total_sum}")

# Open PAM50 spinal cord segmentation
nii_spinalcord = nib.load("../template/PAM50_cord.nii.gz")

# Create labeled spinal cord mask
data_spinalsegments = nii_spinalcord.get_fdata()

# Create a new array for the mid-point voxels, initialized with zeros
data_midpoint = np.zeros(nii_spinalcord.get_fdata().shape)

# Zero values above z_top
data_spinalsegments[:, :, z_top:] = 0

z_segment_top = z_top
i_level = 1
cumulative_percent = 0

# Create labeled segmentation
for level_info in percent_length_segment:
    level_name, level_percent = list(level_info.items())[0]
    cumulative_percent += level_percent
    # Compute the desired absolute position (from the top) based on cumulative percentage
    desired_position = z_top - np.round(length_spinalcord * cumulative_percent / 100)
    # Modify spinal cord mask with spinal segment value
    data_spinalsegments[:, :, int(desired_position):int(z_segment_top)] *= i_level
    # Calculate the mid-point of the current segment and set the voxel value
    midpoint_z = (z_segment_top + desired_position) // 2
    data_midpoint[70, 70, int(midpoint_z)] = i_level
    # Update location of the top of the next segment
    z_segment_top = desired_position
    # Update level
    i_level += 1

# Ensure that the final segment reaches exactly to z_bottom
data_spinalsegments[:, :, z_bottom:int(z_segment_top)] *= (i_level - 1)

# Use dtype UINT8 for labels
data_spinalsegments = data_spinalsegments.astype(np.uint8)
data_midpoint = data_midpoint.astype(np.uint8)

# Save file
nii_spinalsegments = nib.Nifti1Image(data_spinalsegments, nii_spinalcord.affine)
fname_out_levels = "PAM50_spinal_levels.nii.gz"
nib.save(nii_spinalsegments, fname_out_levels)
nii_spinalmidpoint = nib.Nifti1Image(data_midpoint, nii_spinalcord.affine)
fname_out_midpoint = "PAM50_spinal_midpoint.nii.gz"
nib.save(nii_spinalmidpoint, fname_out_midpoint)

print(f"Done! 🎉 \nFiles created: {fname_out_levels}, {fname_out_midpoint}")
