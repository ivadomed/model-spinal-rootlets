import argparse
import os.path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_parser():
    parser = argparse.ArgumentParser(description='Get dice score on Y axis')
    parser.add_argument('-gt', required=True, help='Path to the ground truth')
    parser.add_argument('-pr', required=True, help='Path to the predicted label')
    parser.add_argument('-im', required=True, help='Path to the original image')
    return parser


def get_y_label(file):
    """
    Get mri data into a numpy array
    Args:
        file (str): Path to the nifti file
    Returns:
        image_data (np.array): Image data
    """
    mri = nib.load(file)
    print(f"Orient of {file.split('/')[-1]}: {nib.aff2axcodes(mri.affine)}")
    image_data = mri.get_fdata()
    return image_data


def fn_fp_slice(image_slice, mri, z):
    return 0


def spec_of_slice(gt, lbl, mri, z, print_bool=False):
    """
    From one slice create image with TP, FP, TN, FN voxels and calcul dice score.
    Args:
        gt (np.array): Slice of ground truth
        lbl (np.array): Slice of prediction
        mri (np.array): Slice of original
        z (int): Slice number (z axis)
    Returns:
        dice (float): Dice score for the slice
        cropped_gt_data (np.array): Cropped ground truth around ROI
        colors (np.array): Image with voxel value specific to TP, FP, TN, FN
        cropped_mri_data (np.array): Cropped mri around ROI
    """

    # TODO ADAPT MIN MAX to border
    min_x = min(min(np.where(gt > 0)[0]), min(np.where(lbl > 0)[0])) - 5
    max_x = max(max(np.where(gt > 0)[0]), max(np.where(lbl > 0)[0])) + 5
    min_y = min(min(np.where(gt > 0)[1]), min(np.where(lbl > 0)[1])) - 5
    max_y = max(max(np.where(gt > 0)[1]), max(np.where(lbl > 0)[1])) + 5
    roi_x_start = int(min_x)
    roi_x_end = int(max_x)
    roi_y_start = int(min_y)
    roi_y_end = int(max_y)
    cropped_gt_data = gt[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    cropped_lbl_data = lbl[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    cropped_mri_data = mri[roi_x_start:roi_x_end, roi_y_start:roi_y_end]

    # Create one mask for each prediction type
    tn_mask = np.logical_and(cropped_gt_data == 0, cropped_lbl_data == 0)
    tp_mask = np.logical_and(cropped_gt_data == 1, cropped_lbl_data == 1)
    fp_mask = np.logical_and(cropped_gt_data == 0, cropped_lbl_data == 1)
    fn_mask = np.logical_and(cropped_gt_data == 1, cropped_lbl_data == 0)
    result_img = np.empty(tn_mask.shape)
    result_img[tn_mask] = 0
    result_img[tp_mask] = 3
    result_img[fp_mask] = 2
    result_img[fn_mask] = 1
    printf = f"{z} TP: {len(np.where(result_img == 3)[0])}, FP: {len(np.where(result_img == 2)[0])}, " \
             f"TN: {len(np.where(result_img == 0)[0])}, FN: {len(np.where(result_img == 1)[0])}"
    if print_bool:
        print(printf)
    dice = (2 * len(np.where(result_img == 3)[0])) / (2 * len(np.where(result_img == 3)[0]) +
                                                      len(np.where(result_img == 2)[0]) +
                                                      len(np.where(result_img == 1)[0]))

    return dice, cropped_gt_data, result_img, cropped_mri_data


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    image_path = args.gt
    label_path = args.pr
    mri_path = args.im
    mri_load = nib.load(mri_path)
    mri_dt = mri_load.get_fdata()
    image_dt = get_y_label(image_path)
    z_val_img = np.unique(np.where(image_dt > 0)[2])
    label_dt = get_y_label(label_path)
    z_val_label = np.unique(np.where(label_dt > 0)[2])
    min_val = min(min(z_val_label), min(z_val_img))
    max_val = max(max(z_val_label), max(z_val_img))
    res_dict = {"TP": [[], 0], "FP": [[], 0], "TN": [[], 0], "FN": [[], 0]}
    all_dice = {}
    for z in range(min_val, max_val):
        GT = np.any(z_val_img == z)
        PRED = np.any(z_val_label == z)
        if GT:
            if PRED:
                res_dict["TP"][0].append(z)
                res_dict["TP"][1] += 1
                dice, gt, pred, base = spec_of_slice(image_dt[:, :, z], label_dt[:, :, z], mri_dt[:, :, z], z)
                all_dice[z] = (dice, gt, pred, base)
            else:
                res_dict["FN"][0].append(z)
                res_dict["FN"][1] += 1
        else:
            if PRED:
                res_dict["FP"][0].append(z)
                res_dict["FP"][1] += 1
            else:
                res_dict["TN"][0].append(z)
                res_dict["TN"][1] += 1
    for k in res_dict:
        print(f"{k}:{res_dict[k][1]}")
    print(
        f"Dice on z axis : {(2 * res_dict['TP'][1]) / (2 * res_dict['TP'][1] + res_dict['FP'][1] + res_dict['FN'][1])}")
    print(f"Sensitivity: {res_dict['TP'][1] / (res_dict['TP'][1] + res_dict['FN'][1])}")  # good detection of real case
    print(
        f"Specificity: {res_dict['TN'][1] / (res_dict['TN'][1] + res_dict['FP'][1])}")  # good detection of negative case
    fig_width = len(all_dice) * 2
    fig_height = 4
    fig, axes = plt.subplots(len(all_dice), 3, figsize=(fig_height, fig_width))
    z_dice = []
    for i, slice in enumerate(all_dice):
        axes[i, 0].imshow(all_dice[slice][1], cmap='gray')
        axes[i, 0].axis('off')
        colors_cmap = ['black', 'orange', 'red', 'green']
        # Create a custom colormap using ListedColormap
        custom_cmap = ListedColormap(colors_cmap)
        axes[i, 1].imshow(all_dice[slice][3], cmap='gray')
        axes[i, 1].axis('off')

        # Plot the second slice on the right subplot

        axes[i, 2].imshow(all_dice[slice][2], cmap=custom_cmap)
        axes[i, 1].set_title(f'GT|MRI|Pred,slice: {slice}, Dice: {all_dice[slice][0]}')
        axes[i, 2].axis('off')

        z_dice.append(all_dice[slice][0])

        # Adjust the layout and display the figure
    print(
        f"Mean common slice Dice : {np.mean(z_dice)}")
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig('/Users/theomathieu/Downloads/TP1.pdf', dpi=400, bbox_inches='tight')
