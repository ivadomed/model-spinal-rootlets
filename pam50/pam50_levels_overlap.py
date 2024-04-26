"""
Compute percentage overlap between levels in the PAM50 space obtained using the proposed nnUNet method and PAM50 spinal
levels included in SCT ($SCT_DIR/PAM50/template/PAM50_spinal_levels.nii.gz) based on Frostell et al. 2016.

The script accepts two input arguments:
    - path to nii file with levels obtained using the proposed nnUNet method
    - path to nii file with SCT levels based on Frostell et al. 2016

Then, the script iterates over levels (e.g., 2, 3, etc.) and computes the percentage overlap between the two
segmentations. The percentage overlap is computed as the number of voxels that are present in both segmentations divided
by the total number of voxels in the SCT segmentation. The result is printed to the console.

Authors: Jan Valosek
"""

import os
import argparse

import nibabel as nib
import numpy as np


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Compute percentage overlap between PAM50 levels obtained using the proposed nnUNet method and '
                    'spinal levels included in SCT based on Frostell et al. 2016.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-nnunet',
        required=True,
        help='Paths to the nii files with PAM50 levels obtained using the proposed nnUNet method.'
             'Example: "PAM50_t2_label-rootlet_spinal_levels.nii.gz"'
    )
    parser.add_argument(
        '-sct',
        required=True,
        help='Paths to the nii files with SCT levels based on Frostell et al. 2016.'
             'Example: "template/PAM50_spinal_levels.nii.gz"'
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load the data
    nnunet = nib.load(args.nnunet).get_fdata()
    sct = nib.load(args.sct).get_fdata()

    # Check if the two segmentations have the same shape
    if nnunet.shape != sct.shape:
        raise ValueError('The two segmentations have different shapes.')

    # Initialize the list for the results
    results = list()

    # Get unique levels from the nnUNet segmentation; skip 0 (background)
    levels = np.unique(nnunet)[1:]

    # Iterate over levels
    for level in levels:
        # Get the indices of the voxels that belong to the level in the SCT segmentation
        idx_sct = np.where(sct == level)
        # Get the indices of the voxels that belong to the level in the nnUNet segmentation
        idx_nnunet = np.where(nnunet == level)

        # Compute the percentage overlap
        overlap = len(set(zip(*idx_sct)).intersection(set(zip(*idx_nnunet)))) / len(idx_sct[0])

        results.append(overlap)

    # Print the results
    for level, overlap in enumerate(results, start=2):
        print(f'Level {level}: {overlap * 100:.2f}% overlap')


if __name__ == '__main__':
    main()
