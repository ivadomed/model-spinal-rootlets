import argparse
import nibabel as nib
import numpy as np


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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    image_path = args.gt
    label_path = args.pr
    image_dt = get_y_label(image_path)
    z_val_img = np.unique(np.where(image_dt > 0)[2])
    label_dt = get_y_label(label_path)
    z_val_label = np.unique(np.where(label_dt > 0)[2])
    min_val = min(min(z_val_label),min(z_val_img))
    max_val = max(max(z_val_label),max(z_val_img))
    res_dict = {"TP":[[],0], "FP":[[],0], "TN":[[],0], "FN":[[],0]}

    for z in range(min_val, max_val):
        GT = np.any(z_val_img == z)
        PRED = np.any(z_val_label == z)
        if GT :
            if PRED:
                res_dict["TP"][0].append(z)
                res_dict["TP"][1] += 1
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
        print(f"Z: {z}\t GT: {GT}\t Pred: {PRED}")
    for k in res_dict:
        print(f"{k}:{res_dict[k][1]}")
    print(f"Dice on z axis : {(2*res_dict['TP'][1])/(2*res_dict['TP'][1]+res_dict['FP'][1]+res_dict['FN'][1])}")


