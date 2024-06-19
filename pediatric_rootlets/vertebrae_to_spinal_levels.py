#!/usr/bin/env python
# -*- coding: utf-8
# Compute distance between intervertebral discs and the PMJ along the centerline
#
# For usage, type: python get_distance_pmj_dics -h
# 
# The script requires the SCT conda environment to be activated (because we import the SCT's Image class):
#    source ${SCT_DIR}/python/etc/profile.d/conda.sh
#    conda activate venv_sct
#
# Authors: Katerina Krejci
# Inspired by https://github.com/sct-pipeline/pmj-based-csa/blob/419ece49c81782f23405d89c7b4b15d8e03ed4bd/get_distance_pmj_disc.py

import argparse
import numpy as np
import pandas as pd
from spinalcordtoolbox.image import Image


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute distance between each intervertebral disc and PMJ. Outputs .csv file with results."
                    " | Orientation needs to be RPI")
    parser.add_argument('-centerline', required=True, type=str,
                        help=".csv file of the centerline in the RPI orientation")
    parser.add_argument('-disclabel', required=True, type=str,
                        help="Labels of the intervertebral discs.")
    parser.add_argument('-o', required=False, type=str,
                        default='pmj_disc_distance.csv',
                        help="Output csv filename.")

    return parser


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """
    Compute distance from projected PMJ on centerline and cord centerline.
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


def main():
    parser = get_parser()
    args = parser.parse_args()

    output_data = list()

    # Load disc labels
    fname = args.disclabel
    disc_label = Image(args.disclabel).change_orientation('RPI')
    dim = disc_label.header['pixdim']

    # RPI
    px = dim[1]     # R-L
    py = dim[2]     # P-A
    pz = dim[3]     # S-I

    # Create an array with centerline coordinates
    # Note: the centerline was obtained from an RPI reoriented SC seg using the following script:
    # https://github.com/ivadomed/model-spinal-rootlets/blob/main/inter-rater_variability/02a_rootlets_to_spinal_levels.py
    centerline = np.genfromtxt(args.centerline, delimiter=',')

    # Compute distance from PMJ of the centerline
    arr_distance = get_distance_from_pmj(centerline, centerline[2].argmax(), px, py, pz)

    # Find slices, where are contained disc labels
    discs_slices = np.where(disc_label.data != 0)[-1]
    discs = disc_label.data[np.where(disc_label.data != 0)].tolist()

    # Sort disc_slices descendently
    discs_slices = np.sort(discs_slices)[::-1]
    sorted_disc = np.sort(discs)[::-1]

    # Compute distance from PMJ to each disc
    for idx in range(len(sorted_disc[:8])):
        vertebral_level = idx + 1
        actual_slice_end = discs_slices[idx]
        actual_slice_start = discs_slices[idx+1]
        index_start = int(np.where(arr_distance[1] == actual_slice_start)[0])
        distance_from_pmj_start = float(arr_distance[0][index_start])
        index_end = int(np.where(arr_distance[1] == actual_slice_end)[0])
        distance_from_pmj_end = float(arr_distance[0][index_end])
        output_data.append({'spinal_level': vertebral_level,
                                'fname': fname,
                                'slice_start': actual_slice_start,
                                'slice_end': actual_slice_end,
                                'distance_from_pmj_start': distance_from_pmj_start,
                                'distance_from_pmj_end': distance_from_pmj_end,
                                'height': (distance_from_pmj_start - distance_from_pmj_end)
                                })

    fname_out = fname.replace('.nii.gz', '_pmj_distance_vertebral_disc.csv')

    # Write distances to csv
    df = pd.DataFrame(output_data)
    df.to_csv(fname_out, index=False)
    print(f'CSV file saved in {fname_out}.')


if __name__ == '__main__':
    main()