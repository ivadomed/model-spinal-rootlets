#
# Combine several multi-class (i.e., not binary) segmentations into the reference segmentation using the STAPLE
# algorithm.
#
# Example:
#   python combine_segmentations_from_different_raters.py
#       -i sub-001_T2w_label-rootlet_rater1.nii.gz sub-001_T2w_label-rootlet_rater2.nii.gz sub-001_T2w_label-rootlet_rater3.nii.gz
#       -o sub-001_T2w_label-rootlet_staple.nii.gz
#
# Authors: Jan Valosek
#

import os
import argparse

import numpy as np
import SimpleITK as sitk


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Combine several multi-class (i.e., not binary) segmentations into the reference segmentation '
                    'using the STAPLE algorithm.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        nargs='+',
        help='Paths to the segmentation files separated by spaces. Example: sub-001_T2w_label-rootlet_rater1.nii.gz '
             'sub-001_T2w_label-rootlet_rater2.nii.gz sub-001_T2w_label-rootlet_rater3.nii.gz'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output. Example: sub-001_T2w_label-rootlet_staple.nii.gz'
    )

    return parser


def combine_staple(segmentations, fname_out):
    """
    Combine several segmentations into the reference segmentation using the STAPLE algorithm.

    Note: since the segmentations are multi-class (i.e., not binary), we need to threshold them. We do this for each
    spinal level separately.

    Inspiration: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1STAPLEImageFilter.html#details

    :param segmentations: list of segmentations
    :param fname_out: output file name
    Returns:
    """

    # Initialize the list for reference segmentations for each level
    reference_segmentations_all_levels = list()

    # Loop across spinal levels
    for level in np.unique(sitk.GetArrayFromImage(segmentations[0])):
        # Skip zero (background)
        if level == 0:
            continue

        print(f'Processing level {level} ...')
        segmentations_current_level = list()
        # Binarize the segmentations for a specific level
        for i in range(len(segmentations)):
            segmentations_current_level.append(sitk.BinaryThreshold(
                    segmentations[i],
                    lowerThreshold=int(level),
                    upperThreshold=int(level),
                    insideValue=1,
                    outsideValue=0
                )
            )

        # Combine the segmentations for a specific level using STAPLE
        foregroundValue = 1
        threshold = 0.95
        reference_segmentation_STAPLE_probabilities = sitk.STAPLE(
            segmentations_current_level, foregroundValue
        )

        # Threshold the probabilities
        reference_segmentation_current_level = reference_segmentation_STAPLE_probabilities > threshold

        # Change the reference_segmentation_current_level value to the current level to be able to recreate the
        # multi-class segmentation
        reference_segmentation_current_level = sitk.Cast(
            reference_segmentation_current_level,
            sitk.sitkUInt8
        ) * level

        # Store the combined segmentation for each level into a list
        reference_segmentations_all_levels.append(reference_segmentation_current_level)

    # Now, combine the segmentations for each level back into one multi-class file
    # Initialize the final segmentation
    final_segmentation = sitk.Image(
        segmentations[0].GetSize(),
        sitk.sitkUInt8
    )
    final_segmentation.CopyInformation(segmentations[0])

    for i in range(len(reference_segmentations_all_levels)):
        final_segmentation = sitk.Add(
            final_segmentation,
            reference_segmentations_all_levels[i]
        )

    # Save the reference segmentation
    sitk.WriteImage(
        final_segmentation,
        fname_out
    )
    print(f'Reference segmentation saved to {fname_out}.')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.i]

    # Check if the input path exists
    for fname in full_paths:
        if not os.path.exists(fname):
            raise ValueError('Input path does not exist: {}'.format(fname))

    segmentation_file_names = [
        args.seg1,
        args.seg2,
        args.seg3
    ]

    segmentations = [
        sitk.ReadImage(file_name, sitk.sitkUInt8)
        for file_name in full_paths
    ]

    combine_staple(segmentations, args.o)


if __name__ == '__main__':
    main()
