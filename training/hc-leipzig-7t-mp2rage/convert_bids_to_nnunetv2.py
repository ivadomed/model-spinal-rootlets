"""
This script is used to convert BIDS structure of training data to nnUNetv2 structure for training data.

Usage: python convert_bids_to_nnunetv2.py -i /path/to/data_processed -o /path/to/output_folder
-dataset-name dataset_name -contrast-to-move contrast_name

"""

import os
import shutil
from os.path import exists
import argparse
from argparse import RawTextHelpFormatter

def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script converts BIDS datastructure to nnU-Netv2 datastructure.',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-i',
        required=True,
        help='Path to the data_processed folder.'
    )
    parser.add_argument(
        '-o',
        required=True,
        help='Path to the output folder.'
    )
    parser.add_argument(
        '-dataset-name',
        required=True,
        help='Dataset name for nnUNet structure, that you want to create.'
    )
    parser.add_argument(
        '-contrast-to-move',
        required=True,
        nargs='+',
        help='Name of contrast, that you want to move to nnUNet structure.',
        choices = ['UNIT1', 'UNIT1_neg', 'inv-1_part-mag_MP2RAGE', 'inv-1_part-mag_MP2RAGE_neg',
                   'inv-2_part-mag_MP2RAGE', 'inv-2_part-mag_MP2RAGE_neg', 'T2w']
    )
    return parser


def create_nnunet_datastructure_files(nnunet_data_path, dataset_name):
    '''
    This function is used to create a basic folder structure for the nnUNet datastructure.
    :param nnunet_data_path: It is the path, where datastructure of nnU-Net will be organized.
    :return: Two paths, where images and labels will be stored.
    '''

    if exists(nnunet_data_path) != True:
        os.mkdir(nnunet_data_path)
        os.mkdir(nnunet_data_path + '/nnUNet_raw')
        os.mkdir(nnunet_data_path + f'/nnUNet_raw/{dataset_name}')
        os.mkdir(nnunet_data_path + f'/nnUNet_raw/{dataset_name}/imagesTr')
        os.mkdir(nnunet_data_path + f'/nnUNet_raw/{dataset_name}/labelsTr')

    dst_path_images = f"{nnunet_data_path}/nnUNet_raw/{dataset_name}/imagesTr/"
    dst_path_labels = f"{nnunet_data_path}/nnUNet_raw/{dataset_name}/labelsTr/"

    return (dst_path_images, dst_path_labels)


def convert_bids_to_nnunet_structure(bids_dir_path, nnunet_images_path, nnunet_labels_path, contrast_to_move):
    '''
    This function is used to conversion BIDS datastructure to nnU-Net datastructure.
    :param bids_dir_path: It is the path, where data in BIDS format is stored.
    :param nnunet_images_path: It is the path, where images for nnUNet training will be stored.
    :param nnunet_labels_path: It is the path, where ground truth segmentations for nnUNet training will be stored.
    :return: It returns info, that datastructure for usage of nnU-Net has been stored.
    '''

    # Loop across subjects in data_processed folder
    for actual_path in os.listdir(bids_dir_path):

        # Connect another directory to path if exists
        if os.path.isdir(os.path.join(bids_dir_path, actual_path)) and actual_path.startswith('sub'):
            actual_path = os.path.join(bids_dir_path, actual_path)
            actual_path = os.path.join(actual_path, 'anat')

            # Loop across files to find data for subject and ground truth segmentation
            for file in os.listdir(actual_path):

                # Copy files with specific contrast to nnU-Net datastructure
                if os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "UNIT1.nii.gz") and "UNIT1" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    file_number = file[8:10]
                    contrast = file[11:16]
                    shutil.copy(data_path, nnunet_images_path + f'SCDATA_{file_number}_{contrast}_0000.nii.gz')

                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "UNIT1_neg.nii.gz") and "UNIT1_neg" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    file_number = file[8:10]
                    contrast = file[11:20]
                    shutil.copy(data_path, nnunet_images_path + f'SCDATA_{file_number}_{contrast}_0000.nii.gz')

                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "inv-1_part-mag_MP2RAGE.nii.gz") and "inv-1_part-mag_MP2RAGE" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    file_number = file[8:10]
                    contrast = file[11:33]
                    shutil.copy(data_path, nnunet_images_path + f'SCDATA_{file_number}_{contrast}_0000.nii.gz')

                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "inv-1_part-mag_MP2RAGE_neg.nii.gz") and "inv-1_part-mag_MP2RAGE_neg" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    file_number = file[8:10]
                    contrast = file[11:37]
                    shutil.copy(data_path, nnunet_images_path + f'SCDATA_{file_number}_{contrast}_0000.nii.gz')

                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "inv-2_part-mag_MP2RAGE.nii.gz") and "inv-2_part-mag_MP2RAGE" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    file_number = file[8:10]
                    contrast = file[11:33]
                    shutil.copy(data_path, nnunet_images_path + f'SCDATA_{file_number}_{contrast}_0000.nii.gz')

                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                    "inv-2_part-mag_MP2RAGE_neg.nii.gz") and "inv-2_part-mag_MP2RAGE_neg" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    file_number = file[8:10]
                    contrast = file[11:37]
                    shutil.copy(data_path, nnunet_images_path + f'SCDATA_{file_number}_{contrast}_0000.nii.gz')

                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and "T2w" in file and file.endswith(
                        ".nii.gz") and "T2w" in contrast_to_move:
                    data_path = os.path.join(actual_path, file)
                    shutil.copy(data_path, nnunet_images_path + f'{file[:-7]}_0000.nii.gz')

        # Connect derivatives folder to find ground truth segmentations
        if os.path.isdir(os.path.join(bids_dir_path, actual_path)) and actual_path.startswith('derivatives'):
            actual_path = os.path.join(bids_dir_path, actual_path)
            #actual_path = os.path.join(actual_path, 'labels')
            for path in os.listdir(actual_path):
                if os.path.isdir(os.path.join(actual_path, path)) and path.startswith('sub'):
                    new_path = os.path.join(actual_path, path)
                    new_path = os.path.join(new_path, 'anat')

                    for file in os.listdir(new_path):
                        if os.path.isfile(os.path.join(new_path, file)) and file.startswith("sub") and file.endswith(
                            "UNIT1_label-rootlets_dseg.nii.gz"):
                            gt_path = os.path.join(new_path, file)
                            file_number = file[8:10]

                            # for each contrast,copy the same ground truth segmentation (because the images are
                            # perfectly coregistered)
                            for contrast in contrast_to_move:
                                shutil.copy(gt_path, nnunet_labels_path + f'SCDATA_{file_number}_{contrast}.nii.gz')

    return("Datastructure for nnU-Net has been stored!")


def main():
    parser = get_parser()
    arguments = parser.parse_args()
    bids_dir_path = arguments.i
    nnunet_dir_path = arguments.o
    dataset_name = arguments.dataset_name
    contrast_to_move = arguments.contrast_to_move

    # create basic datastructure for nnU-Net:
    nnunet_folders_healthy_images, nnunet_folders_healthy_labels = create_nnunet_datastructure_files(nnunet_dir_path,
                                                                                                     dataset_name)

    # conversion BIDS nnU-Net DATASTRUCTURE:
    convert_bids_to_nnunet_structure(bids_dir_path, nnunet_folders_healthy_images, nnunet_folders_healthy_labels,
                                     contrast_to_move)


if __name__ == '__main__':
    main()
