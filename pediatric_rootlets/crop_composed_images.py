import nibabel as nib
import sys
import os
from spinalcordtoolbox.image import Image, zeros_like
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
import argparse


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script crops the T2w image (and segmentation and disc labels) to the same size as the top image.',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-i_composed',
        required=True,
        help='Path to the T2w composed image.'
    )
    parser.add_argument(
        '-i_top',
        required=True,
        help='Path to the T2w top image.'
    )
    parser.add_argument(
        '-s',
        required=True,
        help='Path to the segmentation mask.'
    )
    parser.add_argument(
        '-d',
        required=True,
        help='Path to the disc labels.'
    )
    parser.add_argument(
        '-pmj',
        required=True,
        help='Path to the PMJ file.'
    )
    parser.add_argument(
        '-rootlets_seg',
        required=True,
        help='Path to the rootlets segmentation.'
    )

    return parser

def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    composed_image = args.i_composed
    top_image = args.i_top
    seg_mask = args.s
    disc_labels = args.d
    pmj = args.pmj
    rootlets_seg = args.rootlets_seg

    # Load the images and change their orientation to RPI
    img_t2_composed_RPI = Image(composed_image).change_orientation('RPI')
    img_t2_top_RPI = Image(top_image).change_orientation('RPI')

    # Check the shape difference between the two images
    shape_difference_t2 = img_t2_composed_RPI.data.shape[2] - img_t2_top_RPI.data.shape[2]

    # Get name of the cropped images and masks
    crop_composed_image = composed_image.replace('.nii.gz', '_crop.nii.gz')
    crop_seg_mask= seg_mask.replace('.nii.gz', '_crop.nii.gz')
    crop_disc_labels = disc_labels.replace('.nii.gz', '_crop.nii.gz')
    crop_pmj = pmj.replace('.nii.gz', '_crop.nii.gz')
    crop_rootlets_seg = rootlets_seg.replace('.nii.gz', '_crop.nii.gz')

    # Crop the composed image to the same size as the top image
    os.system('sct_crop_image -i ' + composed_image + ' -o ' + crop_composed_image + ' -ymin ' + str(shape_difference_t2))

    # Crop the segmentation mask to the same size as the top image
    os.system('sct_crop_image -i ' + seg_mask + ' -o ' + crop_seg_mask + ' -ymin ' + str(shape_difference_t2))

    # Crop the disc labels to the same size as the top image
    os.system('sct_crop_image -i ' + disc_labels + ' -o ' + crop_disc_labels + ' -ymin ' + str(shape_difference_t2))

    # Crop the PMJ to the same size as the top image
    os.system('sct_crop_image -i ' + pmj + ' -o ' + crop_pmj + ' -ymin ' + str(shape_difference_t2))

    # Crop the rootlets segmentation to the same size as the top image
    os.system('sct_crop_image -i ' + rootlets_seg + ' -o ' + crop_rootlets_seg + ' -ymin ' + str(shape_difference_t2))

if __name__ == '__main__':
    main()