import argparse
import shutil
import glob
import pathlib
import os


def get_parser():
    parser = argparse.ArgumentParser(description='Add label files to derivatives')
    parser.add_argument('--path-image', required=True, help='Path to the image')
    parser.add_argument('--path-dataset', required=True, help='Path to the BIDS Dataset')
    parser.add_argument('--derivatives', required=True, help='Name of derivatives folder')
    return parser


def get_subject_info(file_name):
    """
    Get different information about the current subject

    Args:
        file_name (str): Filename corresponding to the subject image
        contrast_dict (dict): Dictionary, key channel_names from dataset.json

    Returns:
        sub_names (str): Name of the subject. Example: sub-milan002
        ses (str): ses name
        bids_nb (str): subject number in the BIDS dataset
        info[2] (str): Contrast value in BIDS format. Example 0001
        contrast (str): Image contrast (T2w, T1, ...)

    """
    contrast_dict = {
        '0': "T1w",
        '3': "T2w",
    }
    name = file_name.split(".")[0]
    info = name.split("_")
    bids_nb = info[-2]
    contrast = info[-1]
    if len(info) == 4:
        ses = info.pop(1)
    else:
        ses = None
    sub_name = "_".join(info[:-2])
    contrast = contrast.lstrip('0')
    if contrast == '':
        contrast = '0'
    contrast_bids = contrast_dict[contrast]
    print(sub_name, ses, bids_nb, contrast, contrast_bids)
    return sub_name, ses, bids_nb, contrast, contrast_bids


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    image_path = args.path_image
    dataset_path = args.path_dataset
    derivatives = args.derivatives
    print(image_path, dataset_path, derivatives)
    for image in os.listdir(image_path):
        if image.endswith(".nii.gz"):
            print(image, "####")
            sub_name, ses, bids_nb, nnunet_contrast, bids_contrast = get_subject_info(image)
            if ses:
                label_new_dir = os.path.join(dataset_path, 'derivatives', derivatives, sub_name, ses, 'anat')
            else:
                label_new_dir = os.path.join(dataset_path, 'derivatives', derivatives, sub_name, 'anat')
            pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(image_path, image), label_new_dir)


