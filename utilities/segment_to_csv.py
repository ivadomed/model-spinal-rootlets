"""
This script is used to compute the spinal level from the spinal rootlet segmentation.
And compute the height of each spinal level.

Théo MATHIEU
"""
import argparse
import os
import numpy as np
import pandas as pd
import nibabel as nib
import pathlib
from rootlet_to_level import main as rootlet_to_level
from intersect import main as intersect


def get_parser():
    parser = argparse.ArgumentParser(description='Add label files to derivatives')
    parser.add_argument('--image', required=True, help='Path to the original image')
    parser.add_argument('--out', required=True, help='Path to the output folder')
    parser.add_argument('--temp', required=False, help='Path to the temporary folder, default is /tmp/seg_to_csv',
                        default='/tmp/seg_to_csv')
    parser.add_argument('--sc', required=False,
                        help='Path to the spinal cord segmentation. if not provided, the spinal cord will be segmented with SCT')
    parser.add_argument('--centerline', required=False,
                        help='Path to the centerline. if not provided, the centerline will be extracted with SCT')
    parser.add_argument('--rootlet', required=True,
                        help='Path to the spinal rootlet segmentation.')
    parser.add_argument('--spinal_level', required=False,
                        help='Path to the spinal level segmentation. if not provided, the spinal level will be computed from rootlet segmentation')
    parser.add_argument('--pmj', required=False,
                        help="Path to the pmj segmentation. if not provided, the pmj will be computed from SCT")
    return parser


def calc_dist(path_seg, level, path_centerline=None, path_pmj=None):
    """
    Calculate the distance between the pmj and a segmentation point
    Args:
        path_centerline (str): Path to the centerline
        path_pmj (str): Path to the pmj segmentation
        path_seg (str): Path to the segmentation
        level (int): Level to check
    Returns:
        dist (float): Distance between the pmj and the segmentation point
    """
    # TODO get dist along the centerline and convert in mm
    seg = nib.load(path_seg).get_fdata()
    slice = np.unique(np.where((seg >= level - 0.5) & (seg <= level + 0.5))[2])
    if len(slice) == 0:
        return np.nan, np.nan
    # TODO
    start = min(slice)
    end = max(slice)
    return start, end


def get_pmj_position(path_pmj):
    """
    Get the position of the pmj in the image in voxel
    Args:
        path_pmj (str): Path to the pmj segmentation
    Returns:
        pmj_position (int): Position of the pmj in the image in voxel
    """
    pmj_img = nib.load(path_pmj).get_fdata()
    pmj_position = np.where(pmj_img > 0)
    return pmj_position


def main(path_image, path_temp, path_out, df_dict, path_rootlet, path_sc=None, path_centerline=None, rm=False):
    """
    Main function to compute the spinal level from the spinal rootlet segmentation
    Args:
        path_image (str): Path to the original image
        path_temp (str): Path to the temporary folder
        path_out (str): Path to the output folder
        df_dict (dict): Dictionary containing the information about the image
        path_rootlet (str): Path to the spinal rootlet segmentation
        path_sc (str): Path to the spinal cord segmentation. if not provided, the spinal cord will be segmented with SCT
        path_centerline (str): Path to the centerline. if not provided, the centerline will be extracted with SCT
        rm (bool): If True, remove the temporary folder
    Returns:
        df_dict (dict): Dictionary containing the information about the image
        im_name (str): Name of the image without extension
    """
    pathlib.Path(path_temp).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    im_name_ext = path_image.split('/')[-1]
    im_name = path_image.split('/')[-1].split('.')[0]
    size = nib.load(path_image).header.get_zooms()
    try:
        os.symlink(path_image, os.path.join(path_temp, im_name_ext))
    except:
        print("already")
    if path_centerline is None:
        # TODO intercept error
        os.system(
            'sct_get_centerline -i ' + os.path.join(path_temp, im_name_ext) +
            ' -c t2 -o ' + os.path.join(path_temp, im_name + '_centerline.nii.gz -v 0'))
        path_centerline = os.path.join(path_temp, im_name + "_centerline.nii.gz")
    if path_sc is None:
        # TODO intercept error
        os.system('sct_deepseg_sc -i ' + os.path.join(path_temp, im_name_ext) +
                  ' -file_centerline ' + path_centerline + ' -v 0 -c t2 -ofolder ' + path_temp)
        path_sc = os.path.join(path_temp, im_name + "_seg.nii.gz")
    os.system(
        'sct_maths -i ' + path_sc + ' -o ' + os.path.join(path_temp + '/' + im_name + '_dil.nii.gz') + ' -dilate 2')
    image1 = path_temp + '/' + im_name + '_dil.nii.gz'
    path_intersect = os.path.join(path_temp, im_name + '_intersect.nii.gz')
    intersect(path_rootlet, image1, path_intersect)
    rootlet_to_level(path_intersect, path_sc, os.path.join(path_temp, im_name + "_spinal_level.nii.gz"))
    path_spinal_level = os.path.join(path_temp, im_name + "_spinal_level.nii.gz")
    print("All the segmentations are computed. You can now visualize them with the following command:")
    print("\033[92m" + 'fsleyes ' + path_image + ' -cm greyscale ' + path_centerline +
          ' -cm blue ' + path_spinal_level + ' -cm HSV &' + "\033[0m")

    for lvl in range(2, 12):
        df_dict["level"].append(lvl)
        df_dict["sub_name"].append(im_name)
        s_spinal, e_spinal = calc_dist(path_spinal_level, lvl )
        df_dict["spinal_start"].append(s_spinal)
        df_dict["spinal_end"].append(e_spinal)
        df_dict["height"].append((e_spinal - s_spinal) * size[2])
        # s_vertebrae, e_vertebrae = calc_dist(path_centerline, path_pmj, path_vertebrae_level)
        s_vertebrae, e_vertebrae = 0, 1
        df_dict["PMJ_start"].append(s_vertebrae)
        df_dict["PMJ_end"].append(e_vertebrae)
    if rm:
        os.system('rm -rf ' + path_temp + '/*')
    return df_dict, im_name


if __name__ == "__main__":
    args = get_parser().parse_args()
    path_image = args.image
    path_temp = args.temp
    path_out = args.out
    path_sc = args.sc
    path_pmj = args.pmj
    path_centerline = args.centerline
    path_rootlet = args.rootlet
    path_spinal_level = args.spinal_level
    df_dict = {"level": [], "sub_name": [], "spinal_start": [], "spinal_end": [], "height": [],
               "PMJ_start": [], "PMJ_end": []}
    df_dict, im_name = main(path_image, path_temp, path_out, df_dict, path_rootlet, path_sc, path_pmj, path_centerline)
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(path_out, im_name + "_dist.csv"), index=False)
