"""
This script converts nnU-Netv2 datastructure to BIDS datastructure. It copies the predictions to the BIDS structure
and renames the label files to include the dataset name and fold number.

Usage: python convert_nnunetv2_to_bids.py -i /path/to/nnunetv2_results -o /path/to/output_folder

"""

import os
import re
import shutil
import argparse

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_images_to_bids(imagesTs_path, bids_base_dir, subject_id_pattern):
    """
    This function copies the images from the imagesTs folder to the BIDS structure.
    :param imagesTs_path: Folder with images.
    :param bids_base_dir: Folder where the BIDS structure will be created.
    :param subject_id_pattern: Subject ID pattern.
    """
    for file in os.listdir(imagesTs_path):
        if file.endswith(".nii.gz"):
            match = subject_id_pattern.search(file)
            if match:
                subject_id = match.group(1)
                bids_subject_dir = os.path.join(bids_base_dir, f"sub-{subject_id}", "anat")
                ensure_dir_exists(bids_subject_dir)
                file_path = os.path.join(imagesTs_path, file)
                new_image_file_path = os.path.join(bids_subject_dir, file)
                shutil.copy(file_path, new_image_file_path)
                print(f"Copied {file_path} to {new_image_file_path}")

def copy_and_rename_labels(folds, dataset_path, bids_base_dir, subject_id_pattern):
    """
    This function copies the labels from the predictions folders to the BIDS structure and renames them.
    :param folds: Folds in the dataset.
    :param dataset_path: Dataset path.
    :param bids_base_dir: BIDS base directory.
    :param subject_id_pattern: Subject ID pattern.
    """
    for fold in folds:
        fold_path = os.path.join(dataset_path, fold)
        fold = fold[12:]
        for file in os.listdir(fold_path):
            if file.endswith(".nii.gz"):
                match = subject_id_pattern.search(file)
                if match:
                    subject_id = match.group(1)
                    bids_subject_dir = os.path.join(bids_base_dir, f"sub-{subject_id}", "anat")
                    ensure_dir_exists(bids_subject_dir)
                    file_base, _ = os.path.splitext(file)
                    file_base, _ = os.path.splitext(file_base)
                    dataset_id = dataset_path.split('/')[-1][:10]
                    new_label_file_name = f"{file_base}_{dataset_id}_{fold}.nii.gz"
                    file_path = os.path.join(fold_path, file)
                    new_label_file_path = os.path.join(bids_subject_dir, new_label_file_name)
                    shutil.copy(file_path, new_label_file_path)
                    print(f"Copied {file_path} to {new_label_file_path}")

def convert_nnUNet_to_BIDS(nnUNet_raw_dir, bids_base_dir):
    """
    This function converts nnU-Net results to BIDS format.
    :param nnUNet_raw_dir: nnU-Net_raw directory.
    :param bids_base_dir: Directory where the BIDS structure will be created.
    :return:
    """
    subject_id_pattern = re.compile(r'SCDATA_(\d+)')
    for dataset_name in os.listdir(nnUNet_raw_dir):
        dataset_path = os.path.join(nnUNet_raw_dir, dataset_name)
        if os.path.isdir(dataset_path):
            print(f"Processing dataset: {dataset_name}")
            imagesTs_path = os.path.join(dataset_path, 'imagesTs')
            folds = [folder for folder in os.listdir(dataset_path) if folder.startswith("predictions_fold")]

            if os.path.exists(imagesTs_path) and folds:
                copy_images_to_bids(imagesTs_path, bids_base_dir, subject_id_pattern)
                copy_and_rename_labels(folds, dataset_path, bids_base_dir, subject_id_pattern)
            else:
                print(f"Skipping {dataset_name}, as it does not contain imagesTs or predictions folders.")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert nnU-Net results to BIDS format."
    )
    parser.add_argument(
        '-i',
        type=str,
        required=True,
        help="Input directory (where nnUNet_raw folder is stored)"
    )
    parser.add_argument(
        '-o',
        type=str,
        required=True,
        help="Output BIDS directory"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    convert_nnUNet_to_BIDS(args.i, args.o)
    print("Conversion completed.")


if __name__ == "__main__":
    main()