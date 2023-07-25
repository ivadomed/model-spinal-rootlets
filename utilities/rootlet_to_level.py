import argparse
import nibabel as nib
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='Add label files to derivatives')
    parser.add_argument('--rootlet', required=True, help='Path to the spinal rootlet segmentation')
    parser.add_argument('--sc', required=True, help='Path to the spinal cord segmentation')
    parser.add_argument('--out', required=True, help='Path to the output file')
    return parser

def color_sc(sc, value, slice_list):
    """
    Color the spinal cord segmentation depending on the spinal level. By color we mean that we assign a value to the voxel
    Args:
        sc (np.array): Spinal cord segmentation
        value (int): Value to color the spinal cord segmentation
        slice_list (list): List of slices to color
    Returns:
        sc (np.array): Spinal cord segmentation colored
    """
    for slice in slice_list:
        actual = sc[:, :, slice].max()
        if actual == 1.0:
            sc[:, :, slice] = value
        else :
            new = np.mean([actual, value])
            sc[:, :, slice] = new
    return sc

def get_rootlet_slice(rootlet, lvl):
    """
    Get the list of slices where the rootlet is present at a given level.
    Args:
        rootlet (np.array): Rootlet segmentation
        lvl (int): Level to check
    Returns:
        slice_list (list): List of slices where the rootlet is present for a given level
    """
    slice_list = np.unique(np.where(rootlet == lvl)[2])
    return slice_list




def main():
    parser = get_parser()
    args = parser.parse_args()
    path_rootlet = args.rootlet
    path_sc = args.sc
    path_out = args.out
    rootlet = nib.load(path_rootlet).get_fdata()
    orig = nib.load(path_sc)
    sc = orig.get_fdata()
    sc_mask = (sc > 0.1).astype(float)
    for level in range(2,12):
        list_slice = get_rootlet_slice(rootlet, level)
        if len(list_slice) != 0:
            sc = color_sc(sc, level,list_slice)
    nib.save(nib.Nifti1Image(sc * sc_mask, affine=orig.affine, header=orig.header), path_out)



    pass


if __name__ == '__main__':
    main()
