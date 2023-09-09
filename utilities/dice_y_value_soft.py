"""
The script does the following:
    - compute custom f1 score for each spinal level
    - compute mean f1 score across slices for each level
    - compute mean dice score across slices for each level
    - produce PDF report for rootlet segmentation task

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Authors: Théo MATHIEU, Jan Valosek
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from spinalcordtoolbox.image import Image


def get_parser():
    parser = argparse.ArgumentParser(description='Compute f1 score and dice for each level.')
    parser.add_argument('-gt', required=True, help='Path to the ground truth')
    parser.add_argument('-pr', required=True, help='Path to the predicted label')
    parser.add_argument('-im', required=True, help='Path to the original image')
    parser.add_argument('-o', required=True, help='Path to save results')

    return parser


def crop_slice(image_slice, mri):
    """
    Crop the slice around the ROI
    Args:
        image_slice (np.array): Slice of the segmentation image
        mri (np.array): Slice of the original image
    Returns:
        cropped_image_data (np.array): Cropped slice of the segmentation image
        cropped_mri_data (np.array): Cropped slice of the original image
    """
    #TODO adapt min max to border
    min_x = min(np.where(image_slice > 0)[0]) - 5
    max_x = max(np.where(image_slice > 0)[0]) + 5
    min_y = min(np.where(image_slice > 0)[1]) - 5
    max_y = max(np.where(image_slice > 0)[1]) + 5
    roi_x_start = int(min_x)
    roi_x_end = int(max_x)
    roi_y_start = int(min_y)
    roi_y_end = int(max_y)
    cropped_image_data = image_slice[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    cropped_mri_data = mri[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    return cropped_image_data, cropped_mri_data


def process_slice(ground_truth, label, mri):
    """
    For each slice create image with TP, TN, FN, FP voxels and compute f1 score.
    Args:
        ground_truth (np.array): ground truth
        label (np.array): prediction
        mri (np.array): background original
    Returns:
        f1 (float): F1 score for the slice
        cropped_ground_truth_data (np.array): Cropped ground truth around ROI
        colors (np.array): Image with voxel value specific to TP, FP, TN, FN
        cropped_mri_data (np.array): Cropped mri around ROI
    """

    # TODO ADAPT MIN MAX to border
    min_x = min(min(np.where(ground_truth > 0)[0]), min(np.where(label > 0)[0])) - 5
    max_x = max(max(np.where(ground_truth > 0)[0]), max(np.where(label > 0)[0])) + 5
    min_y = min(min(np.where(ground_truth > 0)[1]), min(np.where(label > 0)[1])) - 5
    max_y = max(max(np.where(ground_truth > 0)[1]), max(np.where(label > 0)[1])) + 5
    roi_x_start = int(min_x)
    roi_x_end = int(max_x)
    roi_y_start = int(min_y)
    roi_y_end = int(max_y)
    cropped_ground_truth_data = ground_truth[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    cropped_label_data = label[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    cropped_mri_data = mri[roi_x_start:roi_x_end, roi_y_start:roi_y_end]

    # Create one mask for each prediction type
    # True negative
    tn_mask = np.logical_and(cropped_ground_truth_data == 0, cropped_label_data == 0)
    # Slice positive
    sp_mask = np.logical_and(cropped_ground_truth_data > 0, cropped_label_data > 0)
    # False positive
    fp_mask = np.logical_and(cropped_ground_truth_data == 0, cropped_label_data > 0)
    # False negative
    fn_mask = np.logical_and(cropped_ground_truth_data > 0, cropped_label_data == 0)

    result_img = np.empty(tn_mask.shape)
    result_img[tn_mask] = 0
    result_img[sp_mask] = 3
    result_img[fp_mask] = 2
    result_img[fn_mask] = 1
    # f1 = (2 * SP) / (2 * SP + FN + FP)
    f1 = (2 * len(np.where(result_img == 3)[0])) / (2 * len(np.where(result_img == 3)[0]) +
                                                    len(np.where(result_img == 2)[0]) +
                                                    len(np.where(result_img == 1)[0]))

    return f1, cropped_ground_truth_data, result_img, cropped_mri_data


def compute_dice_slice(ground_truth, prediction):
    """
    Compute dice score for a slice
    Args:
        ground_truth: ground truth
        prediction: prediction
    Returns:
        dice_slice: dice score for the slice
    """
    dice_slice = 2 * np.sum(np.logical_and(ground_truth, prediction)) / (np.sum(ground_truth) + np.sum(prediction))

    return dice_slice


def main():
    parser = get_parser()
    args = parser.parse_args()

    fname_gt = args.gt
    fname_prediction = args.pr
    fname_imame = args.im
    fname_out = args.o

    im_image = Image(fname_imame).change_orientation('RPI')
    im_data = im_image.data

    im_gt = Image(fname_gt).change_orientation('RPI')
    im_gt_data = im_gt.data
    rootlets_levels = np.unique(im_gt_data[np.where(im_gt_data > 0)])

    im_prediction = Image(fname_prediction).change_orientation('RPI')
    im_prediction_data = im_prediction.data

    output_data = list()

    # Loop over the rootlets levels
    for level in rootlets_levels:

        dict_level = dict()

        print(f"Spinal level: {level}")

        # Threshold the GT to keep only the current rootlet level
        gt_level = np.where(im_gt_data == level, 1, 0)
        # Get the slices with rootlets for the GT
        slices_level_gt = np.unique(np.where(gt_level > 0)[2])

        # Threshold the prediction to keep only the current rootlet level
        prediction_level = np.where(im_prediction_data == level, 1, 0)
        # Get the slices with rootlets for the prediction
        slices_level_prediction = np.unique(np.where(prediction_level > 0)[2])

        len_slices_level_gt = len(slices_level_gt)
        len_slices_level_prediction = len(slices_level_prediction)

        if len_slices_level_prediction != 0 or len_slices_level_gt != 0:
            if len_slices_level_prediction == 0:
                slices_level_prediction = [min(slices_level_gt), max(slices_level_gt)]
            elif len_slices_level_gt == 0:
                slices_level_gt = [min(slices_level_prediction), max(slices_level_prediction)]

            all_dice = list()

            min_val = min(min(slices_level_prediction), min(slices_level_gt))
            max_val = max(max(slices_level_prediction), max(slices_level_gt))
            # SP: slice positive - 1 or more voxel similarity (but not 100%) ground truth vs predicted
            # FN: false negative - only ground truth have voxel labeled
            # FP: false positive - only prediction have voxel labeled
            res_dict = {"SP": [[], 0], "TN": [[], 0], "FN": [[], 0], "FP": [[], 0]}
            all_f1 = {"SP": {}, "FN": {}, "FP": {}}

            # Loop across slices for the current rootlet level
            for z_slice in range(min_val, max_val):

                # Compute Dice score for the current slice
                dice_slice = compute_dice_slice(gt_level[:, :, z_slice], prediction_level[:, :, z_slice])
                all_dice.append(dice_slice)

                # Check if the slice is in the GT
                if np.any(slices_level_gt == z_slice):
                    # Check if the slice is in the prediction
                    if np.any(slices_level_prediction == z_slice):
                        res_dict["SP"][0].append(z_slice)
                        res_dict["SP"][1] += 1
                        f1_slice, ground_truth, pred, base = process_slice(gt_level[:, :, z_slice],
                                                                           prediction_level[:, :, z_slice],
                                                                           im_data[:, :, z_slice])
                        all_f1["SP"][z_slice] = (f1_slice, ground_truth, pred, base)
                    else:
                        res_dict["FN"][0].append(z_slice)
                        res_dict["FN"][1] += 1
                        img, base = crop_slice(gt_level[:, :, z_slice], im_data[:, :, z_slice])
                        all_f1["FN"][z_slice] = (0, img, 0, base)
                else:
                    # Check if the slice is in the prediction
                    if np.any(slices_level_prediction == z_slice):
                        res_dict["FP"][0].append(z_slice)
                        res_dict["FP"][1] += 1
                        img, base = crop_slice(prediction_level[:, :, z_slice], im_data[:, :, z_slice])
                        all_f1["FP"][z_slice] = (0, img, 0, base)
                    else:
                        res_dict["TN"][0].append(z_slice)
                        res_dict["TN"][1] += 1
            for key, value in res_dict.items():
                print(f"{key}: {value[1]}")
            # Compute the F1 score for the current rootlet level
            # f1 = (2 * SP) / (2 * SP + FN + FP)
            f1_level = (2 * res_dict['SP'][1]) / (2 * res_dict['SP'][1] + res_dict['FN'][1] + res_dict['FP'][1])
            print(f'f1 score level: {f1_level}')

            for type in ["SP", "FP", "FN"]:
                len_all_f1_type = len(all_f1[type])
                if len_all_f1_type != 0:
                    fig_width = len(all_f1[type]) * 1.5
                    fig_height = 4
                    if type == "SP":
                        fig, axes = plt.subplots(len_all_f1_type, 3, figsize=(fig_height, fig_width))
                        z_slice_f1 = []
                    else:
                        fig, axes = plt.subplots(len_all_f1_type, 3, figsize=(fig_height, fig_width))
                    for i, slice in enumerate(all_f1[type]):
                        if type == "SP":
                            order = 1
                            colors_cmap = ['black', 'orange', 'red', 'green']
                            custom_cmap = ListedColormap(colors_cmap)
                            axes[i, 2].imshow(all_f1["SP"][slice][2], cmap=custom_cmap)
                            axes[i, 2].axis('off')
                            z_slice_f1.append(all_f1[type][slice][0])
                            axes[i, 1].set_title(
                                f'ground_truth|MRI|Pred,slice: {slice}, f1: {all_f1["SP"][slice][0]:.02f}')
                        else:
                            order = 2
                            try:
                                axes[1].set_title(f'Image|MRI,slice: {slice}, type {type}', ha='center')
                                axes[1].axis("off")
                            except:
                                axes[i, 1].set_title(f'Image|MRI,slice: {slice}, type {type}', ha='center')
                                axes[i, 1].axis("off")

                        if len_all_f1_type == 1:
                            axes[0].imshow(all_f1[type][slice][1], cmap='gray')
                            axes[0].axis('off')
                            axes[order].imshow(all_f1[type][slice][3], cmap='gray')
                            axes[order].axis('off')
                        else:
                            axes[i, 0].imshow(all_f1[type][slice][1], cmap='gray')
                            axes[i, 0].axis('off')
                            axes[i, order].imshow(all_f1[type][slice][3], cmap='gray')
                            axes[i, order].axis('off')

                        # Adjust the layout and display the figure
                    plt.subplots_adjust(wspace=0, hspace=0.2)
                    plt.savefig(f"{fname_out}_{type}_{level}.pdf", dpi=400, bbox_inches='tight')
                else:
                    # print(f"not possible for {type}")
                    pass

            # Compute mean f1 score across slices for the current rootlet level
            mean_f1 = np.mean(z_slice_f1)
            print(f"Mean common F1 : {mean_f1}")

            # Compute mean Dice score across slices for the current rootlet level
            mean_dice = np.mean(all_dice)
            print(f"Mean Dice : {mean_dice}")
            print("")

            dict_level = {'f1_level': f1_level,
                          'mean_f1_across_slices': mean_f1,
                          'mean_dice_across_slices': mean_dice,
                          'SP': res_dict['SP'][1],
                          'FP': res_dict['FP'][1],
                          'TN': res_dict['TN'][1],
                          'FN': res_dict['FN'][1],
                          }

        # Note: **dict_level is used to unpack the key-value pairs from the metrics dictionary
        output_data.append({'level': level, **dict_level})

    # Create a pandas DataFrame from the parsed data
    df = pd.DataFrame(output_data)

    # Save the DataFrame to a CSV file
    fname_out = f'{fname_out}_f1_and_dice_scores.csv'
    df.to_csv(fname_out, index=False)
    print(f'Parsed data saved to {fname_out}')


if __name__ == '__main__':
    main()
