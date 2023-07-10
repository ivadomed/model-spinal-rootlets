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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    image_path = args.path_image
    dataset_path = args.path_dataset
    derivatives = args.derivatives
    for image in os.listdir(image_path):
        if image.endswith(".nii.gz"):
            sub_name = image.split("_")[0]
            filename = f"{sub_name}_T2w_spinalroot-manual.nii.gz"
            label_new_dir = os.path.join(dataset_path, 'derivatives', derivatives, sub_name, 'anat')
            pathlib.Path(label_new_dir).mkdir(parents=True, exist_ok=True)
            print(f"{os.path.join(image_path, image)} copy to {label_new_dir}")
            shutil.copy2(os.path.join(image_path, image), os.path.join(label_new_dir, filename))
