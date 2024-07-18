"""
This script zeros the T2w image (label files (SC seg, PMJ, ...)) below a specific intervertebral disc
(specified by "-x").

The script requires the SCT conda environment to be activated (because we import the SCT's Image class):
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct
"""

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
        description='The script zeros the rootlets segmentation below a specific intervertebral disc (specified by "-x").',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-rootlets-seg',
        required=True,
        help='Path to the rootlets segmentation.'
    )
    parser.add_argument(
        '-d',
        required=True,
        help='Path to the disc labels.'
    )
    parser.add_argument(
        '-x',
        required=True,
        type=int,
        help='Disc level to zero the image below.'
    )

    return parser


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    rootlets_seg = args.rootlets_seg
    disc_labels = args.d
    x = int(args.x)

    # Load the images and change orientation to RPI
    original_orientation_rootlets = Image(rootlets_seg).orientation
    disc_labels_RPI= Image(disc_labels).change_orientation('RPI')
    rootlets_seg_RPI = Image(rootlets_seg).change_orientation('RPI')

    # find the coordinates, where is the disc label x
    disc_label_x = np.where(disc_labels_RPI.data == x)

    # Put zeros to the rootlets segmentation under the disc level x
    rootlets_seg_RPI.data[:, :, :disc_label_x[2][0]] = 0
    rootlets_seg_modif = rootlets_seg.replace('.nii.gz', '_modif.nii.gz')
    rootlets_seg_RPI = Image(rootlets_seg_RPI).change_orientation(original_orientation_rootlets)

    # Save the modified segmentation
    rootlets_seg_RPI.save(rootlets_seg_modif)


if __name__ == '__main__':
    main()
