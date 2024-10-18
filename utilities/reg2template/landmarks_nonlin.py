#!/usr/bin/env python
# -*- coding: utf-8
# TODO
#
# For usage, type: python landmarks_nonlin.py -h

# Authors: Sandrine Bédard

import argparse
import numpy as np
import nibabel as nib
import os
import numpy as np
from scipy.optimize import minimize


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute distance between each intervertebral disc and nerve rootlet, and PMJ to nerve rootlets. Outputs .csv file with results. | Orientation needs to be RPI")
    parser.add_argument('-src', required=True, type=str,
                        help=".csv file of the centerline.")
    parser.add_argument('-dest', required=False, type=str,
                        help="Labels of the intervertebral discs.")
    parser.add_argument('-o', required=False, type=str,
                        default='pmj_disc_distance.csv',
                        help="Output csv filename.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    src = nib.load(args.src)
    src_np = np.array(src.get_fdata())
    # Change zeros to nan to ignore during mean
    src_np[src_np == 0] = np.nan
    src_np_mean = np.zeros_like(src_np)
   # print(src_np_mean)
    for slices in range(src_np.shape[2]):
        if np.isnan(src_np[:,:,slices]).all():
            mean_np = [[0]]
        else:
            mean_np = np.nanmean(src_np[:,:,slices],keepdims=True)
        src_np_mean[:,:,slices] = np.broadcast_to(mean_np, src_np[:,:,slices].shape)
    nii_mean_warp= nib.Nifti1Image(src_np_mean, src.affine)
    fname_out_levels = args.o
    nib.save(nii_mean_warp, fname_out_levels)


    

if __name__ == '__main__':
    main()