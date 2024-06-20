import os
from spinalcordtoolbox.image import Image, zeros_like
from argparse import RawTextHelpFormatter
import argparse
import numpy as np


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
    parser.add_argument(
        '-x',
        required=True,
        help='Exact disc level, where to crop the images.'
    )

    return parser

def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    composed_image = args.i_composed
    seg_mask = args.s
    disc_labels = args.d
    pmj = args.pmj
    rootlets_seg = args.rootlets_seg
    x= int(args.x)

    # Load the images and change orientation to RPI
    disc_labels_RPI= Image(disc_labels).change_orientation('RPI')

    # find the coordinates, where is the disc label x
    disc_label_x = np.where(disc_labels_RPI.data == x)

    # Set the level, where to crop the images
    cropping_level = disc_label_x[2][0]

    # Get name of the cropped images and masks
    crop_composed_image = composed_image.replace('.nii.gz', '_crop.nii.gz')
    crop_seg_mask= seg_mask.replace('.nii.gz', '_crop.nii.gz')
    crop_disc_labels = disc_labels.replace('.nii.gz', '_crop.nii.gz')
    crop_pmj = pmj.replace('.nii.gz', '_crop.nii.gz')
    crop_rootlets_seg = rootlets_seg.replace('.nii.gz', '_crop.nii.gz')

    # Crop the composed image
    os.system('sct_crop_image -i ' + composed_image + ' -o ' + crop_composed_image + ' -ymin ' + str(cropping_level))

    # Crop the segmentation mask
    os.system('sct_crop_image -i ' + seg_mask + ' -o ' + crop_seg_mask + ' -ymin ' + str(cropping_level))

    # Crop the disc labels
    os.system('sct_crop_image -i ' + disc_labels + ' -o ' + crop_disc_labels + ' -ymin ' + str(cropping_level))

    # Crop the PMJ
    os.system('sct_crop_image -i ' + pmj + ' -o ' + crop_pmj + ' -ymin ' + str(cropping_level))

    # Crop the rootlets segmentation
    os.system('sct_crop_image -i ' + rootlets_seg + ' -o ' + crop_rootlets_seg + ' -ymin ' + str(cropping_level))

if __name__ == '__main__':
    main()