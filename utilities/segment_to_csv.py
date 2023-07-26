import argparse
import os
import shutil

import numpy as np
import pandas as pd
import nibabel as nib
import pathlib
from rootlet_to_level import main as rootlet_to_level


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


def calc_dist(path_centerline, path_pmj, path_seg, level):
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
    # TODO
    start = 0
    end = 1
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


def get_sizing(path_image):
    return nib.load(path_image).header.get_zooms()


def main():
    args = get_parser().parse_args()
    path_image = args.image
    im_name_ext = path_image.split('/')[-1]
    im_name = path_image.split('/')[-1].split('.')[0]
    path_temp = args.temp
    path_out = args.out
    path_sc = args.sc
    path_pmj = args.pmj
    path_centerline = args.centerline
    path_rootlet = args.rootlet
    path_spinal_level = args.spinal_level
    pathlib.Path(path_temp).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    df_dict = {"level": [], "pmj_position": [], "spinal_start": [], "spinal_end": [],
               "vertebrae_start": [], "vertebrae_end": []}
    try:
        os.symlink(path_image, os.path.join(path_temp, im_name_ext))
    except:
        print("already")
    if path_centerline is None:
        os.system(
            'sct_get_centerline -i ' + os.path.join(path_temp, im_name_ext) + ' -c t2 -o ' + os.path.join(path_temp,
                                                                                                          f"{im_name}_centerline.nii.gz"))
        path_centerline = os.path.join(path_temp, f"{im_name}_centerline.nii.gz")
    if path_sc is None:
        os.system('sct_deepseg_sc -i ' + os.path.join(path_temp,
                                                      im_name_ext) + ' -file_centerline ' + path_centerline + ' -c t2 -ofolder ' + path_temp)
        path_sc = os.path.join(path_temp, f"{im_name}_seg.nii.gz")
    if path_spinal_level is None:
        rootlet_to_level(path_rootlet, path_sc, os.path.join(path_temp, f"{im_name}_spinal_level.nii.gz"))
        path_level = os.path.join(path_temp, f"{im_name}_spinal_level.nii.gz")
    if path_pmj is None:
        os.system('sct_detect_pmj -i ' + os.path.join(path_temp,
                                                      im_name_ext) + ' -c t2 -ofolder ' + path_temp + ' -o ' + f"{im_name}_pmj.nii.gz")
        path_pmj = os.path.join(path_temp, f"{im_name}_pmj.nii.gz")

    print("All the segmentations are computed. You can now visualize them with the following command:")
    print(
        "\033[92m" + "fsleyes /tmp/seg_to_csv/sub-brnoUhb01_085_0000.nii.gz -cm greyscale /tmp/seg_to_csv/sub-brnoUhb01_085_0000_pmj.nii.gz -cm red /tmp/seg_to_csv/sub-brnoUhb01_085_0000_centerline.nii.gz -cm blue /tmp/seg_to_csv/sub-brnoUhb01_085_0000_spinal_level.nii.gz -cm HSV &" + "\033[0m")

    pmj_pos = get_pmj_position(path_pmj)
    for lvl in range(2, 12):
        df_dict["level"].append(lvl)
        df_dict["pmj_position"].append(pmj_pos)
        s_spinal, e_spinal = calc_dist(path_centerline, path_pmj, path_spinal_level, lvl)
        df_dict["spinal_start"].append(s_spinal)
        df_dict["spinal_end"].append(e_spinal)
        # s_vertebrae, e_vertebrae = calc_dist(path_centerline, path_pmj, path_vertebrae_level)
        s_vertebrae, e_vertebrae = 0, 1
        df_dict["vertebrae_start"].append(s_vertebrae)
        df_dict["vertebrae_end"].append(e_vertebrae)
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(path_out, f"{im_name}_dist.csv"), index=False)


if __name__ == "__main__":
    main()
