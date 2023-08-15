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
import subprocess
from datetime import datetime


def get_parser():
    # TODO More information about SCT command used ? (version, commit, ...)
    parser = argparse.ArgumentParser(description="""Create a csv file with different information about spinal level: 
                                                 Spinal level start and end slice,
                                                 Spinal level height,
                                                 Distance spinal level-PMJ """)
    parser.add_argument('--image', required=True, help='Path to the original image')
    parser.add_argument('--out', required=True, help='Path to the output folder')
    parser.add_argument('--temp', required=False, help='Path to the temporary folder, default is /tmp/seg_to_csv',
                        default='/tmp/seg_to_csv')
    parser.add_argument('--sc', required=False,
                        help='Path to the spinal cord segmentation. '
                             'If not provided, the spinal cord will be segmented with SCT')
    parser.add_argument('--centerline', '-c', required=False,
                        help='Path to the centerline. If not provided, the centerline will be extracted with SCT')
    parser.add_argument('--spinal-nerve', '-sn', required=True,
                        help='Path to the spinal nerve segmentation.')
    parser.add_argument('--spinal-level', '-sl', required=False,
                        help='Path to the spinal level segmentation. '
                             'If not provided, the spinal level will be computed from rootlet segmentation')
    parser.add_argument('--pmj', required=False,
                        help="Path to the pmj segmentation. If not provided, the pmj will be computed from SCT")
    parser.add_argument('--dilate', '-d', required=False, type=str, default='1',
                        help='Dilation of the spinal cord segmentation')
    parser.add_argument('--rm', required=False, type=bool, default=True, help='delete temp folder')
    return parser


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """

    """
    length = 0
    arr_length = [0]
    for i in range(z_index, 0, -1):
        distance = np.sqrt(((centerline_points[i, 0] - centerline_points[i - 1, 0]) * px) ** 2 +
                           ((centerline_points[i, 1] - centerline_points[i - 1, 1]) * py) ** 2 +
                           ((centerline_points[i, 2] - centerline_points[i - 1, 2]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[:z_index + 1, 2]), axis=0)
    return arr_length


def pmj_dist(centerline_dist, start, end):
    if not np.isnan(start):
        dist_start = - centerline_dist[0, start]
    else:
        dist_start = np.nan
    if not np.isnan(end):
        dist_end = - centerline_dist[0, end]
    else:
        dist_end = np.nan
    return dist_start, dist_end


def calc_height(path_seg, level):
    """
    Calculate the distance between the pmj and a segmentation point
    Args:
        path_centerline (str): Path to the centerline
        path_pmj (str): Path to the pmj segmentation
        path_seg (str): Path to the segmentation
        level (int): Level to check
    Returns:
        start (int): Start slice of the level
        end (int): End slice of the level
    """
    seg = nib.load(path_seg).get_fdata()
    slice = np.unique(np.where((seg >= level - 0.5) & (seg <= level + 0.5))[2])
    if len(slice) == 0:
        return np.nan, np.nan
    end = min(slice)
    start = max(slice)
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


def main(path_image, path_temp, path_out, df_dict, path_nerve, path_sc=None, path_pmj=False, path_centerline=None,
         path_spinal_level=None,dilate=1, rm=False):
    """
    Main function to get different values 
    Args:
        path_image (str): Path to the original image
        path_temp (str): Path to the temporary folder
        path_out (str): Path to the output folder
        df_dict (dict): Dictionary containing the information about the image
        path_nerve (str): Path to the spinal nerve segmentation
        path_sc (str): Path to the spinal cord segmentation. if not provided, the spinal cord will be segmented with SCT
        path_centerline (str): Path to the centerline. if not provided, the centerline will be extracted with SCT
        rm (bool): If True, remove the temporary folder
    Returns:
        df_dict (dict): Dictionary containing the information about the image
        im_name (str): Name of the image without extension
    """
    log_file_path = os.path.join(path_out, f"log_{datetime.now().strftime('%Y-%m-%d-%H:%M')}.txt")
    log_file = open(log_file_path, 'w')
    pathlib.Path(path_temp).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    im_root = path_image.split('/')[-1]
    im_name = path_image.split('/')[-1].split('.')[0]
    size = nib.load(path_image).header.get_zooms()
    try:
        os.symlink(path_image, os.path.join(path_temp, im_root))
    except:
        print("Image already in temporary folder")
    if path_centerline is None:
        print("Centerline not provided, extracting it with SCT")
        log_file.write("Centerline not provided, extracting it with SCT\n")
        # TODO intercept error
        command_output = subprocess.run(['sct_get_centerline', '-i', os.path.join(path_temp, im_root), '-c', 't2', '-o',
                                         os.path.join(path_temp, im_name + '_centerline.nii.gz')],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
        log_file.write(command_output.stdout + '\n\n')
        print("error centerline", command_output.stderr)
        path_centerline = os.path.join(path_temp, im_name + "_centerline.nii.gz")
    if path_sc is None:
        # TODO intercept error
        log_file.write("Spinal cord not provided, extracting it with SCT\n")
        command_output = subprocess.run(['sct_deepseg_sc', '-i', os.path.join(path_temp, im_root), '-file_centerline',
                                         path_centerline, '-c', 't2', '-ofolder', path_temp],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
        log_file.write(command_output.stdout + '\n\n')
        #print("error SC", command_output.stderr)
        path_sc = os.path.join(path_temp, im_name + "_seg.nii.gz")
    if path_spinal_level is None:
        log_file.write("Spinal level not provided, extracting it with SCT\n")
        command_output = subprocess.run(
            ['sct_maths', '-i', path_sc, '-o', os.path.join(path_temp + '/' + im_name + '_dil.nii.gz'),
             '-dilate', dilate], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log_file.write(command_output.stdout + '\n\n')
        #print("error dilate", command_output.stderr)
        image1 = path_temp + '/' + im_name + '_dil.nii.gz'
        path_intersect = os.path.join(path_temp, im_name + '_intersect.nii.gz')
        intersect(path_nerve, image1, path_intersect)
        rootlet_to_level(path_intersect, path_sc, os.path.join(path_temp, im_name + "_spinal_level.nii.gz"))
        path_spinal_level = os.path.join(path_temp, im_name + "_spinal_level.nii.gz")
    print("All the segmentations are computed. You can now visualize them with the following command:")
    print("\033[92m" + 'fsleyes ' + path_image + ' -cm greyscale ' + path_centerline +
          ' -cm blue ' + path_spinal_level + ' -cm HSV &' + "\033[0m")
    log_file.write("All the segmentations are computed. You can now visualize them with the following command:\n")
    log_file.write("\033[92m" + 'fsleyes ' + path_image + ' -cm greyscale ' + path_centerline +
                   ' -cm blue ' + path_spinal_level + ' -cm HSV &' + "\033[0m\n")
    log_file.close()
    z_index = get_pmj_position(path_pmj)[2][0]
    print("PMJ position:", z_index)
    centerline_csv = np.genfromtxt(path_centerline,delimiter=',')
    distance_centerline = get_distance_from_pmj(centerline_csv, z_index, size[0], size[1], size[2])

    for lvl in range(2, 12):
        df_dict["level"].append(lvl)
        df_dict["sub_name"].append(im_name)
        s_spinal, e_spinal = calc_height(path_spinal_level, lvl)
        df_dict["spinal_start"].append(s_spinal)
        df_dict["spinal_end"].append(e_spinal)
        df_dict["height"].append((s_spinal - e_spinal) * size[2])
        s_vertebrae, e_vertebrae = pmj_dist(distance_centerline, s_spinal, e_spinal)
        df_dict["PMJ_start"].append(s_vertebrae)
        df_dict["PMJ_end"].append(e_vertebrae)
    if rm:
        print("Removing temporary files")
        os.system('rm -rf ' + path_temp + '/*')
    return df_dict, im_name


if __name__ == "__main__":
    args = get_parser().parse_args()
    path_image = args.image
    path_temp = args.temp
    path_out = args.out
    path_sc = args.sc
    path_pmj = args.pmj
    dilate = args.dilate
    path_centerline = args.centerline
    path_nerve = args.spinal_nerve
    path_spinal_level = args.spinal_level
    rm = args.rm
    all_path_optional = [path_temp, path_sc, path_pmj, path_centerline, path_nerve, path_spinal_level]
    all_path_str = ["path_temp", "path_sc", "path_pmj", "path_centerline", "path_nerve", "path_spinal_level"]
    for i, path in enumerate(all_path_optional):
        if path is None:
            print(f"Path to {all_path_str[i]} is not provided. It will be use to compute.")
    df_dict = {"level": [], "sub_name": [], "spinal_start": [], "spinal_end": [], "height": [],
               "PMJ_start": [], "PMJ_end": []}
    df_dict, im_name = main(path_image, path_temp, path_out, df_dict, path_nerve, path_sc, path_pmj, path_centerline,
                            path_spinal_level, dilate, rm)
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(path_out, im_name + f"_pmj-dist-dilate{dilate}.csv"), index=False)
