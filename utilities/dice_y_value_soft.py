"""
Script to compute custom metric and produce pdf report for rootlet segmentation task.

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Authors: Théo MATHIEU, Jan Valosek
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from spinalcordtoolbox.image import Image


def get_parser():
    parser = argparse.ArgumentParser(description='Get f1 score on Y axis')
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


def tp_slice(ground_truth, label, mri):
    """
    From one slice create image with TP, FP, TN, FN voxels and compute f1 score.
    Args:
        ground_truth (np.array): F1 of ground truth
        label (np.array): Slice of prediction
        mri (np.array): Slice of original
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
    tn_mask = np.logical_and(cropped_ground_truth_data == 0, cropped_label_data == 0)
    tp_mask = np.logical_and(cropped_ground_truth_data > 0, cropped_label_data > 0)
    fp_mask = np.logical_and(cropped_ground_truth_data == 0, cropped_label_data > 0)
    fn_mask = np.logical_and(cropped_ground_truth_data > 0, cropped_label_data == 0)
    result_img = np.empty(tn_mask.shape)
    result_img[tn_mask] = 0
    result_img[tp_mask] = 3
    result_img[fp_mask] = 2
    result_img[fn_mask] = 1
    f1 = (2 * len(np.where(result_img == 3)[0])) / (2 * len(np.where(result_img == 3)[0]) +
                                                    len(np.where(result_img == 2)[0]) +
                                                    len(np.where(result_img == 1)[0]))

    return f1, cropped_ground_truth_data, result_img, cropped_mri_data


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

    for level in rootlets_levels:
        print(f"#### - Spinal level: {level} - ####")

        # Threshold the GT to keep only the current rootlet level
        gt_slice = np.where(im_gt_data == level, 1, 0)
        # Get the slices with rootlets for the GT
        slices_level_gt = np.unique(np.where(gt_slice > 0)[2])

        # Threshold the prediction to keep only the current rootlet level
        prediction_slice = np.where(im_prediction_data == level, 1, 0)
        # Get the slices with rootlets for the prediction
        slices_level_prediction = np.unique(np.where(prediction_slice > 0)[2])

        len_slices_level_gt = len(slices_level_gt)
        len_slices_level_prediction = len(slices_level_prediction)

        if len_slices_level_prediction != 0 or len_slices_level_gt != 0:
            if len_slices_level_prediction == 0:
                slices_level_prediction = [min(slices_level_gt), max(slices_level_gt)]
            elif len_slices_level_gt == 0:
                slices_level_gt = [min(slices_level_prediction), max(slices_level_prediction)]
            min_val = min(min(slices_level_prediction), min(slices_level_gt))
            max_val = max(max(slices_level_prediction), max(slices_level_gt))
            res_dict = {"TP": [[], 0], "FP": [[], 0], "TN": [[], 0], "FN": [[], 0]}
            all_f1 = {"TP": {}, "FN": {}, "FP": {}}
            for z_slice in range(min_val, max_val):
                ground_truth = np.any(slices_level_gt == z_slice)
                PRED = np.any(slices_level_prediction == z_slice)
                if ground_truth:
                    if PRED:
                        res_dict["TP"][0].append(z_slice)
                        res_dict["TP"][1] += 1
                        f1, ground_truth, pred, base = tp_slice(gt_slice[:, :, z_slice], prediction_slice[:, :, z_slice],
                                                                im_data[:, :, z_slice])
                        all_f1["TP"][z_slice] = (f1, ground_truth, pred, base)
                    else:
                        res_dict["FN"][0].append(z_slice)
                        res_dict["FN"][1] += 1
                        img, base = crop_slice(gt_slice[:, :, z_slice], im_data[:, :, z_slice])
                        all_f1["FN"][z_slice] = (0, img, 0, base)
                else:
                    if PRED:
                        res_dict["FP"][0].append(z_slice)
                        res_dict["FP"][1] += 1
                        img, base = crop_slice(prediction_slice[:, :, z_slice], im_data[:, :, z_slice])
                        all_f1["FP"][z_slice] = (0, img, 0, base)
                    else:
                        res_dict["TN"][0].append(z_slice)
                        res_dict["TN"][1] += 1
            for i, k in enumerate(res_dict):
                if i == 1 or i == 3:
                    print(f"{k}:{res_dict[k][1]}")
                else:
                    print(f"{k}:{res_dict[k][1]}", end="\t")
            f1_z_slice = f"Z-axis F1 score : {(2 * res_dict['TP'][1]) / (2 * res_dict['TP'][1] + res_dict['FP'][1] + res_dict['FN'][1])}"
            print(f1_z_slice)

            for type in ["TP", "FP", "FN"]:
                len_all_f1_type = len(all_f1[type])
                if len_all_f1_type != 0:
                    fig_width = len(all_f1[type]) * 1.5
                    fig_height = 4
                    if type == "TP":
                        fig, axes = plt.subplots(len_all_f1_type, 3, figsize=(fig_height, fig_width))
                        z_slice_f1 = []
                    else:
                        fig, axes = plt.subplots(len_all_f1_type, 3, figsize=(fig_height, fig_width))
                    for i, slice in enumerate(all_f1[type]):
                        if type == "TP":
                            order = 1
                            colors_cmap = ['black', 'orange', 'red', 'green']
                            custom_cmap = ListedColormap(colors_cmap)
                            axes[i, 2].imshow(all_f1["TP"][slice][2], cmap=custom_cmap)
                            axes[i, 2].axis('off')
                            z_slice_f1.append(all_f1[type][slice][0])
                            axes[i, 1].set_title(
                                f'ground_truth|MRI|Pred,slice: {slice}, f1: {all_f1["TP"][slice][0]:.02f}')
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
                    plt.savefig(f"{fname_out}{type}_{level}.pdf", dpi=400, bbox_inches='tight')
                else:
                    # print(f"not possible for {type}")
                    pass
            mean_f1 = f"Mean common F1 : {np.mean(z_slice_f1)}"
            print(mean_f1, end="\n\n")

            table = [
                ['ground_truth/Pred', 'P', 'N'],
                ['P', f"{res_dict['TP'][1]}", f"{res_dict['FN'][1]}"],
                ['N', f"{res_dict['FP'][1]}", f"{res_dict['TN'][1]}"]
            ]

            # Open the file in write mode
            with open(f'{fname_out}result_{level}.log', 'w') as file:
                # Iterate over each row in the table
                for row in table:
                    # Join the elements of the row with tab ('\t') as separator
                    # and write it to the file
                    file.write('\t'.join(row))
                    file.write('\n')  # Write a newline character to move to the next row
                file.write(f1_z_slice)
                file.write('\n')
                file.write(mean_f1)


if __name__ == '__main__':
    main()
