import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_parser():
    parser = argparse.ArgumentParser(description='Get dice score on Y axis')
    parser.add_argument('-gt', required=True, help='Path to the ground truth')
    parser.add_argument('-pr', required=True, help='Path to the predicted label')
    return parser


def get_y_label(file):
    mri = nib.load(file)
    print(nib.aff2axcodes(mri.affine))
    image_data = mri.get_fdata()
    return image_data


def spec_of_slice(gt, lbl, z, graph=False):
    gt_slice = gt[:, :, z]
    lbl_slice = lbl[:, :, z]
    # TODO ADAPT MIN MAX to border
    min_x = min(min(np.where(gt_slice > 0)[0]), min(np.where(lbl_slice > 0)[0])) - 5
    max_x = max(max(np.where(gt_slice > 0)[0]), max(np.where(lbl_slice > 0)[0])) + 5
    min_y = min(min(np.where(gt_slice > 0)[1]), min(np.where(lbl_slice > 0)[1])) - 5
    max_y = max(max(np.where(gt_slice > 0)[1]), max(np.where(lbl_slice > 0)[1])) + 5
    roi_x_start = int(min_x)  # Replace with the starting X coordinate of the ROI
    roi_x_end = int(max_x)  # Replace with the ending X coordinate of the ROI
    roi_y_start = int(min_y)  # Replace with the starting Y coordinate of the ROI
    roi_y_end = int(max_y)

    cropped_gt_data = gt_slice[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    cropped_lbl_data = lbl_slice[roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    comparison = np.equal(cropped_gt_data, cropped_lbl_data)

    colors = np.where(comparison, 0, 1)
    x, y = np.where(cropped_gt_data > 0)
    colors[x, y] = 2
    # print(f"{z} TP: {len(np.where(colors==1)[0])}, FP+FN: {len(np.where(colors==2)[0])}, TN: {len(np.where(colors==0)[0])}")
    Dice = (2 * len(np.where(colors == 2)[0])) / (2 * len(np.where(colors == 2)[0]) + len(np.where(colors == 1)[0]))
    if graph:
        fig, axes = plt.subplots(1, 2)
        # Plot the first slice on the left subplot
        axes[0].imshow(cropped_gt_data, cmap='gray')
        axes[0].set_title('Slice 1')
        axes[0].axis('off')

        # Plot the second slice on the right subplot
        axes[1].imshow(cropped_lbl_data, cmap='gray')
        axes[1].set_title('Slice 2')
        axes[1].axis('off')
        colors_cmap = ['black', 'red', 'green']
        # Create a custom colormap using ListedColormap
        custom_cmap = ListedColormap(colors_cmap)
        axes[1].imshow(colors, cmap=custom_cmap)

        # Adjust the layout and display the figure
        plt.tight_layout()
        plt.suptitle(f"z : {z}")
        plt.show()

    return Dice, cropped_gt_data, colors


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    image_path = args.gt
    label_path = args.pr
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
                Dice, gt, pred = spec_of_slice(image_dt, label_dt, z)
                all_dice[z] = (Dice, gt, pred)
                print(f"Z: {z}\t GT: {GT}\t Pred: {PRED}\t Dice slice: {Dice}")
            else:
                res_dict["FN"][0].append(z)
                res_dict["FN"][1] += 1
                print(f"Z: {z}\t GT: {GT}\t Pred: {PRED}")
        else:
            if PRED:
                res_dict["FP"][0].append(z)
                res_dict["FP"][1] += 1
            else:
                res_dict["TN"][0].append(z)
                res_dict["TN"][1] += 1
            print(f"Z: {z}\t GT: {GT}\t Pred: {PRED}")
    for k in res_dict:
        print(f"{k}:{res_dict[k][1]}")
    print(
        f"Dice on z axis : {(2 * res_dict['TP'][1]) / (2 * res_dict['TP'][1] + res_dict['FP'][1] + res_dict['FN'][1])}")
    print(f"Sensitivity: {res_dict['TP'][1] / (res_dict['TP'][1] + res_dict['FN'][1])}")  # good detection of real case
    print(
        f"Specificity: {res_dict['TN'][1] / (res_dict['TN'][1] + res_dict['FP'][1])}")  # good detection of negative case
    fig_width = len(all_dice) * 2
    fig_height = 4
    fig, axes = plt.subplots(len(all_dice), 2, figsize=(fig_height, fig_width))
    for i, slice in enumerate(all_dice):
        axes[i, 0].imshow(all_dice[slice][1], cmap='gray')
        axes[i, 0].axis('off')

        # Plot the second slice on the right subplot
        colors_cmap = ['black', 'red', 'green']
        # Create a custom colormap using ListedColormap
        custom_cmap = ListedColormap(colors_cmap)
        axes[i, 1].imshow(all_dice[slice][2], cmap=custom_cmap)
        axes[i, 1].set_title(f'Pred {z},Dice: {all_dice[slice][0]}')
        axes[i, 1].axis('off')
    #plt.colorbar(ticks=np.linspace(0, 2, 3))

        # Adjust the layout and display the figure
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig('/Users/theomathieu/Downloads/output.pdf', dpi=400, bbox_inches='tight')
