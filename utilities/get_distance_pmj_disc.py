#!/usr/bin/env python
# -*- coding: utf-8
# Compute distance between nerve rootlets and PMJ, nerve rootlets and discs along centerline
#
# For usage, type: python get_distance_pmj_dics -h

# Authors: Sandrine Bédard

import argparse
import csv
import numpy as np
import nibabel as nib
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute distance between each intervertebral disc and nerve rootlet, and PMJ to nerve rootlets. Outputs .csv file with results. | Orientation needs to be RPI")
    parser.add_argument('-centerline', required=True, type=str,
                        help=".csv file of the centerline.")
    parser.add_argument('-disclabel', required=True, type=str,
                        help="Labels of the intervertebral discs.")
    parser.add_argument('-spinalroots', required=True, type=str,
                        help="Labels of the spinal nerve rootlets.")
    parser.add_argument('-subject', required=True, type=str,
                        help="Subject ID")
    parser.add_argument('-o', required=False, type=str,
                        default='pmj_disc_distance.csv',
                        help="Output csv filename.")

    return parser


def save_Nifti1(data, original_image, filename):
    empty_header = nib.Nifti1Header()
    image = nib.Nifti1Image(data, original_image.affine, empty_header)
    nib.save(image, filename)


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

    disc_label = nib.load(args.disclabel)
    dim = disc_label.header['pixdim']

    nerve_label = nib.load(args.spinalroots)

    px = dim[0]
    py = dim[1]
    pz = dim[2]
    # Create an array with centerline coordinates
    centerline = np.genfromtxt(args.centerline, delimiter=',')
    # Get C2-C3 disc coordinate
    # Compute distance from PMJ of the centerline
    arr_distance = get_distance_from_pmj(centerline, centerline[2].argmax(), px, py, pz)
    # Get discs labels
    discs_index = np.where(disc_label.get_fdata() != 0)[-1]
    discs = disc_label.get_fdata()[np.where(disc_label.get_fdata() != 0)].tolist()
    nerve_index = np.where(nerve_label.get_fdata() != 0)[-1]
    nerves = nerve_label.get_fdata()[np.where(nerve_label.get_fdata() != 0)]
    nerves = list(map(int, nerves))
    for j in range(len(nerves)):
        nerve = nerves[j]
        if nerve in discs:
            disc = discs[discs.index(nerve)]
            disc_index_corr = np.abs(centerline[2] - discs_index[discs.index(nerve)]).argmin()  # centerline doesn't necessarly start at the index 0 if the segmentation is incomplete
            nerve2_index_corr = np.abs(centerline[2] - nerve_index[j]).argmin()
            distance_disc_nerve = arr_distance[:, disc_index_corr][0] - arr_distance[:, nerve2_index_corr][0]  # Disc to nerve
        else:
            distance_disc_nerve = np.NaN
            disc = np.NaN
        subject = args.subject
        fname_out = args.o
        distance_pmj_nerve = np.NaN
        if not os.path.isfile(fname_out):
            with open(fname_out, 'w') as csvfile:
                header = ['Subject', 'Nerve', 'Disc', 'Distance - PMJ (mm)', 'Distance - Disc (mm)']
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
        with open(fname_out, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            line = [subject, nerve, disc, distance_pmj_nerve, distance_disc_nerve]
            spamwriter.writerow(line)


    for i in range(len(nerves)):
        # Get the index of centerline array disc
        nerve = nerves[i]
        nerve_index_corr = np.abs(centerline[2] - nerve_index[i]).argmin()
        distance_pmj_nerve = arr_distance[:, nerve_index_corr][0]
        subject = args.subject
        fname_out = args.o
        disc = np.NaN
        distance_disc_nerve = np.NaN
        if not os.path.isfile(fname_out):
            with open(fname_out, 'w') as csvfile:
                header = ['Subject', 'Nerve', 'Disc', 'Distance - PMJ (mm)', 'Distance - Disc (mm)']
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
        with open(fname_out, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            line = [subject, nerve, disc, distance_pmj_nerve, distance_disc_nerve]
            spamwriter.writerow(line)


if __name__ == '__main__':
    main()
