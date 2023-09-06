"""
The script does the following:
    - project the nerve rootlets on the spinal cord segmentation to obtain spinal levels
    - compute the distance between the pontomedullary junction (PMJ) and the start and end of the spinal level (PMJ
    label is required)

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Authors: Jan Valosek, Theo Mathieu
"""

import os
import argparse
import numpy as np
import pandas as pd

from argparse import RawTextHelpFormatter
from spinalcordtoolbox.image import Image, zeros_like


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script does the following:'
                    '\n\t- project the nerve rootlets on the spinal cord segmentation to obtain spinal levels'
                    '\n\t- compute the distance between the pontomedullary junction (PMJ) and the start and end of '
                    'the spinal level (PMJ label is required)',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-i',
        required=True,
        help='Path to the spinal nerve rootlet segmentation.'
    )
    parser.add_argument(
        '-s',
        required=True,
        help='Path to the spinal cord segmentation.'
    )
    parser.add_argument(
        '-pmj',
        required=False,
        help='Path to the pontomedullary junction (PMJ) label. If provided, the script computes the distance between '
             'the PMJ and the start and end of the spinal level.'
    )
    parser.add_argument(
        '-dilate',
        required=False,
        type=int,
        help='Size of spinal cord segmentation dilation in pixels. Large number leads to "longer" spinal levels. '
             'Default: 2.',
        default=2,
    )

    return parser


def get_centerline_from_pmj(fname_seg, fname_pmj):
    """
    Generate extrapolated centerline from pontomedullary junction (PMJ).
    :param fname_seg: spinal cord segmentation
    :param fname_pmj: PMJ label
    :return: fname_centerline: path to the CSV file with the extrapolated centerline
    """
    # Inspiration: https://github.com/sct-pipeline/pmj-based-csa/blob/e362536a7bef17c0e151830cf777fb51fd09cb87/process_data.sh#L157-L160
    # Note: -v 2 is used to generate the extrapolated centerline CSV
    os.system('sct_process_segmentation -i ' + fname_seg + ' -pmj ' + fname_pmj + ' -pmj-distance 50 -v 2')
    # Remove unnecessary png files and csa.csv
    os.system('rm -f *.png')
    os.system('rm -f csa.csv')

    fname_centerline = fname_seg.replace('.nii.gz', '_centerline_extrapolated.csv')

    return fname_centerline


def intersect_seg_and_rootlets(im_rootlets, fname_seg, fname_rootlets, dilate_size):
    """
    Intersect the spinal cord segmentation and the spinal nerve rootlet segmentation.
    :param im_rootlets: Image object of the spinal nerve rootlet segmentation
    :param fname_seg: path to the spinal cord segmentation
    :param fname_rootlets: path to the spinal nerve rootlet segmentation
    :param dilate_size: size of spinal cord segmentation dilation in pixels
    :return: fname_intersect: path to the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    """

    # Dilate the SC segmentation using sct_maths
    fname_seg_dil = fname_seg.replace('.nii.gz', '_dil.nii.gz')
    os.system('sct_maths -i ' + fname_seg + ' -o ' + fname_seg_dil + ' -dilate ' + str(dilate_size))

    # Load the dilated SC segmentation
    im_seg_dil = Image(fname_seg_dil).change_orientation('RPI')
    im_seg_dil_data = im_seg_dil.data

    # Intersect the rootlets and the dilated SC segmentation
    intersect_data = im_rootlets.data * im_seg_dil_data

    # Save the intersection using the Image class
    im_intersect = zeros_like(im_rootlets)
    im_intersect.data = intersect_data
    fname_intersect = fname_rootlets.replace('.nii.gz', '_intersect.nii.gz')
    im_intersect.save(fname_intersect)
    print(f'Intersection saved in {fname_intersect}.')

    return fname_intersect


def project_rootlets_to_segmentation(im_rootlets, im_seg, im_intersect, rootlets_levels, fname_rootlets):
    """"
    Project the nerve rootlets intersection on the spinal cord segmentation
    :param im_rootlets: Image object of the spinal nerve rootlet segmentation
    :param im_seg: Image object of the spinal cord segmentation
    :param im_intersect: Image object of the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    :param rootlets_levels: list of the spinal nerve rootlets levels
    :param fname_rootlets: path to the spinal nerve rootlet segmentation
    :return: fname_spinal_levels: path to the spinal levels segmentation
    :return: start_end_slices: list of the spinal levels start and end slices
    """
    im_spinal_levels_data = np.copy(im_seg.data)

    start_end_slices = dict()

    # Loop across the rootlets levels
    for level in rootlets_levels:
        # Get the list of slices where the level is present
        slices_list = np.unique(np.where(im_intersect.data == level)[2])
        # Skip the level if it is not present in the intersection
        if len(slices_list) != 0:
            min_slice = min(slices_list)
            max_slice = max(slices_list)
            start_end_slices[level] = {'start': min_slice, 'end': max_slice}
            # Color the SC segmentation with the level
            im_spinal_levels_data[:, :, min_slice:max_slice][im_seg.data[:, :, min_slice:max_slice] == 1] = level

    # Set zero to the slices with no intersection
    im_spinal_levels_data[im_spinal_levels_data == 1] = 0

    # Save the projection using the Image class
    im_spinal_levels = zeros_like(im_rootlets)
    im_spinal_levels.data = im_spinal_levels_data
    fname_spinal_levels = fname_rootlets.replace('.nii.gz', '_spinal_levels.nii.gz')
    im_spinal_levels.save(fname_spinal_levels)
    print(f'Spinal levels file saved in {fname_spinal_levels}.')

    return fname_spinal_levels, start_end_slices


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """
    Compute distance from projected pontomedullary junction (PMJ) on centerline and cord centerline.
    Inspiration: https://github.com/sct-pipeline/pmj-based-csa/blob/419ece49c81782f23405d89c7b4b15d8e03ed4bd/get_distance_pmj_disc.py#L40-L60
    :param centerline_points: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param z_index: z index PMJ on the centerline.
    :param px: x pixel size.
    :param py: y pixel size.
    :param pz: z pixel size.
    :return: nd-array: distance from PMJ and corresponding indexes.
    """
    length = 0
    arr_length = [0]
    for i in range(z_index, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2 +
                           ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2 +
                           ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2][:z_index + 1]), axis=0)

    return arr_length


def pmj_dist(centerline_dist, start, end):
    """
    Compute the distance between the pontomedullary junction (PMJ) and the start and end of the spinal level
    :param centerline_dist: distance between the PMJ and the centerline
    :param start: start slice of the spinal level
    :param end: end slice of the spinal level
    :return: dist_start: distance between the PMJ and the start of the spinal level
    :return: dist_end: distance between the PMJ and the end of the spinal level
    """
    if not np.isnan(start):
        dist_start = centerline_dist[0, start]
    else:
        dist_start = np.nan
    if not np.isnan(end):
        dist_end = centerline_dist[0, end]
    else:
        dist_end = np.nan
    return dist_start, dist_end


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    fname_rootlets = args.i
    fname_seg = args.s
    dilate_size = args.dilate

    # Load input images using the SCT Image class
    im_rootlets = Image(fname_rootlets).change_orientation('RPI')
    im_seg = Image(fname_seg).change_orientation('RPI')

    # Intersect the rootlets and the SC segmentation
    fname_intersect = intersect_seg_and_rootlets(im_rootlets, fname_seg, fname_rootlets, dilate_size)

    # Load the intersection
    im_intersect = Image(fname_intersect).change_orientation('RPI')

    # Get unique values in the rootlets segmentation larger than 0
    rootlets_levels = np.unique(im_rootlets.data[np.where(im_rootlets.data > 0)])

    # Project the nerve rootlets intersection on the spinal cord segmentation to obtain spinal levels
    fname_spinal_levels, start_end_slices = project_rootlets_to_segmentation(im_rootlets, im_seg, im_intersect,
                                                                             rootlets_levels, fname_rootlets)

    if args.pmj:
        fname_pmj = args.pmj
        im_pmj = Image(fname_pmj).change_orientation('RPI')

        # Generate extrapolated centerline from PMJ
        fname_centerline = get_centerline_from_pmj(fname_seg, fname_pmj)

        # Load CSV file with centerline coordinates generated by the previous command as an array
        centerline = np.genfromtxt(fname_centerline, delimiter=',')
        # Compute distance from PMJ of the centerline
        arr_distance = get_distance_from_pmj(centerline, centerline[2].argmax(), im_pmj.dim[4], im_pmj.dim[5],
                                             im_pmj.dim[6])

    output_data = list()
    for level in rootlets_levels:
        print(f'Processing level {level}...')

        # Compute the distance between the PMJ and the start and end of the spinal level
        dist_start, dist_end = pmj_dist(arr_distance, start_end_slices[level]['start'], start_end_slices[level]['end'])

        output_data.append({'spinal_level': level,
                            'fname': fname_rootlets,
                            'slice_start': start_end_slices[level]['start'],
                            'slice_end': start_end_slices[level]['end'],
                            'distance_from_pmj_start': dist_start,
                            'distance_from_pmj_end': dist_end,
                            'height': dist_start - dist_end
                            })

    # Create a pandas DataFrame
    df = pd.DataFrame(output_data)

    # Save the DataFrame as a CSV file
    fname_out = fname_rootlets.replace('.nii.gz', '.csv')
    df.to_csv(fname_out, index=False)
    print(f'CSV file saved in {fname_out}.')


if __name__ == '__main__':
    main()
