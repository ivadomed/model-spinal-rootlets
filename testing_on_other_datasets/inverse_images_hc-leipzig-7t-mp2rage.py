
"""
This script creates the negative of the input image.

Usage:
    python inverse_images_hc-leipzig-7t-mp2rage.py -i <input_image> -o <output_image>

Author: Katerina Krejci
"""

import os
import argparse
import numpy as np

from argparse import RawTextHelpFormatter
from spinalcordtoolbox.image import Image, zeros_like


def get_parser():
    parser = argparse.ArgumentParser(
        description='The script does the following:'
                    '\n\t- create negative of the input image',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-i',
        required=True,
        help='Path to the original image.'
    )
    parser.add_argument(
        '-o',
        required=True,
        help='Path to the output image, where the negative image will be saved'
    )

    return parser


def main(args):

    # Load the image
    im = Image(args.i).change_orientation('RPI')

    # Create negative of the image
    im_inv = zeros_like(im)
    im_inv.data = np.max(im.data) - im.data

    # Save the inverted image
    im_inv.save(args.o)

    print(f'Inverted image saved at: {args.o}')


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    main(arguments)