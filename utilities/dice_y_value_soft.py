import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_parser():
    parser = argparse.ArgumentParser(description='Get f1 score on Y axis')
    parser.add_argument('-gt', required=True, help='Path to the ground truth')
    parser.add_argument('-pr', required=True, help='Path to the predicted label')
    parser.add_argument('-im', required=True, help='Path to the original image')
    parser.add_argument('-o', required=True, help='Path to save results')
    parser.add_argument('-v', required=True, nargs="+", type=int, help='Possible values')
    return parser


def nifti2array_sel(file, value):
    """
    Get mri data into a numpy array
    Args:
        file (str): Path to the nifti file
    Returns:
        image_data (np.array): Image data
    """
    mri = nib.load(file)
    print(f"Orient of {file.split('/')[-1]}: {nib.aff2axcodes(mri.affine)}")
    image_data = np.where(mri.get_fdata() == value, 1, 0)
    return image_data


def nifti2array(file):
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


def fn_fp_slice(image_slice, mri):
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
    From one slice create image with TP, FP, TN, FN voxels and calcul f1 score.
    Args:
        ground_truth (np.array): Slice of ground truth
        label (np.array): Slice of prediction
        mri (np.array): Slice of original
    Returns:
        f1 (float): Dice score for the slice
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
    dice = (2 * len(np.where(result_img == 3)[0])) / (2 * len(np.where(result_img == 3)[0]) +
                                                      len(np.where(result_img == 2)[0]) +
                                                      len(np.where(result_img == 1)[0]))

    return dice, cropped_ground_truth_data, result_img, cropped_mri_data


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    image_path = args.gt
    label_path = args.pr
    mri_path = args.im
    values = args.v
    out = args.o
    mri_load = nib.load(mri_path)
    mri_dt = mri_load.get_fdata()
    for val in values:
        print(val)
        image_dt = nifti2array_sel(image_path, val)
        z_slice_val_img = np.unique(np.where(image_dt > 0)[2])
        label_dt = nifti2array_sel(label_path, val)
        z_slice_val_label = np.unique(np.where(label_dt > 0)[2])
        min_val = min(min(z_slice_val_label), min(z_slice_val_img))
        max_val = max(max(z_slice_val_label), max(z_slice_val_img))
        res_dict = {"TP": [[], 0], "FP": [[], 0], "TN": [[], 0], "FN": [[], 0]}
        all_f1 = {"TP": {}, "FN": {}, "FP": {}}
        for z_slice in range(min_val, max_val):
            ground_truth = np.any(z_slice_val_img == z_slice)
            PRED = np.any(z_slice_val_label == z_slice)
            if ground_truth:
                if PRED:
                    res_dict["TP"][0].append(z_slice)
                    res_dict["TP"][1] += 1
                    f1, ground_truth, pred, base = tp_slice(image_dt[:, :, z_slice], label_dt[:, :, z_slice],
                                                            mri_dt[:, :, z_slice])
                    all_f1["TP"][z_slice] = (f1, ground_truth, pred, base)
                else:
                    res_dict["FN"][0].append(z_slice)
                    res_dict["FN"][1] += 1
                    img, base = fn_fp_slice(image_dt[:, :, z_slice], mri_dt[:, :, z_slice])
                    all_f1["FN"][z_slice] = (0, img, 0, base)
            else:
                if PRED:
                    res_dict["FP"][0].append(z_slice)
                    res_dict["FP"][1] += 1
                    img, base = fn_fp_slice(label_dt[:, :, z_slice], mri_dt[:, :, z_slice])
                    all_f1["FP"][z_slice] = (0, img, 0, base)
                else:
                    res_dict["TN"][0].append(z_slice)
                    res_dict["TN"][1] += 1
        for k in res_dict:
            print(f"{k}:{res_dict[k][1]}")
        Dice_z_slice = f"Z-axis F1 score : {(2 * res_dict['TP'][1]) / (2 * res_dict['TP'][1] + res_dict['FP'][1] + res_dict['FN'][1])}"
        #sensi = f"Sensitivity: {res_dict['TP'][1] / (res_dict['TP'][1] + res_dict['FN'][1])}"  # good detection of real case
        #speci = f"Specificity: {res_dict['TN'][1] / (res_dict['TN'][1] + res_dict['FP'][1])}"  # good detection of negative case

        for type in ["TP", "FP", "FN"]:
            try:
                fig_width = len(all_f1[type]) * 2
                fig_height = 4
                if type == "TP":
                    fig, axes = plt.subplots(len(all_f1[type]), 3, figsize=(fig_height, fig_width))
                    z_slice_f1 = []
                else:
                    fig, axes = plt.subplots(len(all_f1[type]), 2, figsize=(fig_height, fig_width))

                for i, slice in enumerate(all_f1[type]):
                    axes[i, 0].imshow(all_f1[type][slice][1], cmap='gray')
                    axes[i, 0].axis('off')
                    axes[i, 1].imshow(all_f1[type][slice][3], cmap='gray')
                    axes[i, 1].axis('off')
                    if type == "TP":
                        colors_cmap = ['black', 'orange', 'red', 'green']
                        custom_cmap = ListedColormap(colors_cmap)
                        axes[i, 2].imshow(all_f1["TP"][slice][2], cmap=custom_cmap)
                        axes[i, 1].set_title(f'ground_truth|MRI|Pred,slice: {slice}, f1: {all_f1["TP"][slice][0]}')
                        axes[i, 2].axis('off')
                        z_slice_f1.append(all_f1[type][slice][0])
                    else:
                        axes[i, 1].set_title(f'Image|MRI,slice: {slice}, type {type}')

                    # Adjust the layout and display the figure
                plt.subplots_adjust(wspace=0, hspace=0.2)
                plt.savefig(f"{out}{type}_{val}.pdf", dpi=400, bbox_inches='tight')
            except:
                print(f"not possible for {type}")
        mean_dice = f"Mean common F1 : {np.mean(z_slice_f1)}"

        table = [
            ['ground_truth/Pred', 'P', 'N'],
            ['P', f"{res_dict['TP'][1]}", f"{res_dict['FN'][1]}"],
            ['N', f"{res_dict['FP'][1]}", f"{res_dict['TN'][1]}"]
        ]

        # Open the file in write mode
        with open(f'{out}result_{val}.log', 'w') as file:
            # Iterate over each row in the table
            for row in table:
                # Join the elements of the row with tab ('\t') as separator
                # and write it to the file
                file.write('\t'.join(row))
                file.write('\n')  # Write a newline character to move to the next row
            file.write(Dice_z_slice)
            #file.write('\n')
            #file.write(sensi)
            #file.write('\n')
            #file.write(speci)
            file.write('\n')
            file.write(mean_dice)
