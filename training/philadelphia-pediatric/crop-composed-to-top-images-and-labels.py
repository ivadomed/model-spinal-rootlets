"""
This script is used to crop the composed images and labels to the same dimensions as the top images and then
convert the BIDS datastructure to nnU-Net datastructure.

This script is necessary because we need to crop composed images and labels to top images (covering C spine) to be able to use them
for nnU-Net training.

Usage: python crop-composed-to-top-images-and-labels.py -i /path/to/data_processed -o /path/to/output_folder -dataset 107

The script requires the SCT conda environment to be activated (because we import the SCT's Image class):
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct
"""

import os
import shutil
from os.path import exists
import argparse
from argparse import RawTextHelpFormatter
from spinalcordtoolbox.image import Image
import subprocess
import glob


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script crops the composed images to the top-images size and converts BIDS datastructure to '
                    'nnU-Netv2 datastructure.',
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
        '-dataset',
        required=True,
        help='Dataset number - e.g. 107.'
    )
    return parser


def create_nnunet_datastructure_files(nnunet_data_path, dataset_number):
    '''
    This function is used to create a basic folder structure for the nnUNet datastructure.
    :param nnunet_data_path: It is the path, where datastructure of nnU-Net will be organized.
    :return: Two paths, where images and labels will be stored.
    '''

    if exists(nnunet_data_path) != True:
        os.mkdir(nnunet_data_path)
        os.mkdir(nnunet_data_path + '/nnUNet_raw')
        os.mkdir(nnunet_data_path + f'/nnUNet_raw/Dataset{dataset_number}')
        os.mkdir(nnunet_data_path + f'/nnUNet_raw/Dataset{dataset_number}/imagesTr')
        os.mkdir(nnunet_data_path + f'/nnUNet_raw/Dataset{dataset_number}/labelsTr')

    dst_path_images = f"{nnunet_data_path}/nnUNet_raw/Dataset{dataset_number}/imagesTr/"
    dst_path_labels = f"{nnunet_data_path}/nnUNet_raw/Dataset{dataset_number}/labelsTr/"

    return (dst_path_images, dst_path_labels)


def convert_bids_to_nnunet_structure(bids_dir_path, nnunet_images_path, nnunet_labels_path):
    """
    This function is used to conversion BIDS datastructure to nnU-Net datastructure.
    :param bids_dir_path: It is the path, where data in BIDS format is stored.
    :param nnunet_images_path: It is the path, where images for nnUNet training will be stored.
    :param nnunet_labels_path: It is the path, where ground truth segmentations for nnUNet training will be stored.
    """

    # Loop across subjects in data_processed folder
    for actual_path in os.listdir(bids_dir_path):

        # Connect another directory to path if exists
        if os.path.isdir(os.path.join(bids_dir_path, actual_path)) and actual_path.startswith('sub'):
            actual_path = os.path.join(bids_dir_path, actual_path)
            actual_path = os.path.join(actual_path, 'anat')

            files = glob.glob(actual_path + '/*')

            # get list of files that contains ending "rootlets_dseg.nii.gz" or "rootlets_dseg_modif.nii.gz"
            rootlets_dseg_files = [file for file in files if file.endswith("rootlets_dseg.nii.gz") or
                                   file.endswith("rootlets_dseg_modif.nii.gz")]

            # Loop across files to find top image for getting dimensions, in which we will crop composed images
            for file in os.listdir(actual_path):
                if os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "top_run-1_T2w.nii.gz"):
                    data_path = os.path.join(actual_path, file)
                    image = Image(data_path).change_orientation('RPI')
                    image_size = image.dim
                    break

            for file in os.listdir(actual_path):
                file_number = file[4:7]
                if os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "dseg_modif.nii.gz"):
                    data_path = os.path.join(actual_path, file)
                    label_composed = Image(data_path).change_orientation('RPI')
                    label_composed_size = label_composed.dim

                    # crop composed label according to dimension from top image
                    command = [
                        'sct_crop_image',
                        '-i', data_path,
                        '-o', f'{data_path[:-7]}_crop.nii.gz',
                        '-ymax', str(label_composed_size[2]), '-ymin', str(label_composed_size[2] - image_size[2])]

                    subprocess.run(command, capture_output=True, text=True)

                    # change orientation of cropped label to LPI and save
                    cropped_label_composed = f'{data_path[:-7]}_crop.nii.gz'
                    lpi_crop_segm = Image(cropped_label_composed).change_orientation('LPI')
                    lpi_crop_segm.save(cropped_label_composed)

                    # copy cropped label to nnU-Net datastructure
                    shutil.copy(cropped_label_composed, nnunet_labels_path + f'PEDIATRIC_{file_number}.nii.gz')


                # if manually modified rootlets segmentation is not present, use original one
                elif os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "dseg.nii.gz") and (os.path.join(actual_path, file[:-7] + "_modif.nii.gz") not in
                                            rootlets_dseg_files and os.path.join(actual_path,
                                                                                 file[:-5] + "_modif.nii.gz") not in
                                            rootlets_dseg_files):
                    data_path = os.path.join(actual_path, file)
                    label_composed = Image(data_path).change_orientation('RPI')
                    label_composed_size = label_composed.dim

                    # crop composed label according to dimension from top image
                    command = [
                        'sct_crop_image',
                        '-i', data_path,
                        '-o', f'{data_path[:-7]}_crop.nii.gz',
                        '-ymax', str(label_composed_size[2]), '-ymin', str(label_composed_size[2] - image_size[2])]

                    subprocess.run(command, capture_output=True, text=True)

                    data_path_crop = f'{data_path[:-7]}_crop.nii.gz'
                    lpi_crop_segm = Image(data_path_crop).change_orientation('LPI')
                    lpi_crop_segm.save(data_path_crop)

                    # copy cropped label to nnU-Net datastructure
                    shutil.copy(data_path_crop, nnunet_labels_path + f'PEDIATRIC_{file_number}.nii.gz')

                if os.path.isfile(os.path.join(actual_path, file)) and file.startswith("sub") and file.endswith(
                        "composed_T2w.nii.gz"):
                    data_path = os.path.join(actual_path, file)

                    t2w_image = Image(data_path).change_orientation('RPI')
                    t2w_image_size = t2w_image.dim

                    # crop image accordin to dimension from top image
                    command = [
                        'sct_crop_image',
                        '-i', data_path,
                        '-o', f'{data_path[:-7]}_crop.nii.gz',
                        '-ymax', str(t2w_image_size[2]), '-ymin', str(t2w_image_size[2] - image_size[2])]

                    subprocess.run(command, capture_output=True, text=True)

                    image_path_crop = f'{data_path[:-7]}_crop.nii.gz'
                    lpi_crop_img = Image(image_path_crop).change_orientation('LPI')
                    lpi_crop_img.save(image_path_crop)

                    # copy cropped image to nnU-Net datastructure
                    shutil.copy(image_path_crop, nnunet_images_path + f'PEDIATRIC_{file_number}_0000.nii.gz')

                if os.path.isfile(os.path.join(actual_path, file)) and (file.startswith("sub-125") or
                        file.startswith("sub-107")) and file.endswith("T2w.nii.gz"):
                    data_path = os.path.join(actual_path, file)

                    image_label = Image(data_path).change_orientation('RPI')
                    image_label_size = image_label.dim

                    # crop image accordin to dimension from top image
                    command = [
                        'sct_crop_image',
                        '-i', data_path,
                        '-o', f'{data_path[:-7]}_crop.nii.gz',
                        '-ymax', str(image_label_size[2]), '-ymin', str(image_label_size[2] - image_size[2])]

                    subprocess.run(command, capture_output=True, text=True)

                    image_path_crop = f'{data_path[:-7]}_crop.nii.gz'
                    lpi_crop_img = Image(image_path_crop).change_orientation('LPI')
                    lpi_crop_img.save(image_path_crop)

                    # copy cropped image to nnU-Net datastructure
                    shutil.copy(image_path_crop, nnunet_images_path + f'PEDIATRIC_{file_number}_0000.nii.gz')


def main():
    parser = get_parser()
    arguments = parser.parse_args()
    bids_dir_path = arguments.i
    nnunet_dir_path = arguments.o
    dataset_number = arguments.dataset

    # create basic datastructure for nnU-Net:
    nnunet_folders_healthy_images, nnunet_folders_healthy_labels = create_nnunet_datastructure_files(nnunet_dir_path,
                                                                                                     dataset_number)

    # conversion BIDS nnU-Net DATASTRUCTURE:
    convert_bids_to_nnunet_structure(bids_dir_path, nnunet_folders_healthy_images, nnunet_folders_healthy_labels)


if __name__ == '__main__':
    main()
