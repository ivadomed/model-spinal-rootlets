import os
import argparse
import numpy as np
from nibabel import load, Nifti1Image

study_to_sub = {
    "13696": 1,
    "13697": 2,
    "13746": 3,
    "13747": 4,
    "13755": 5,
    "13757": 6,
    "14348": 7,
    "14350": 8,
    "14406": 9,
    "14407": 10,
    "14411": 11,
    "14492": 12,
    "14493": 13,
    "14693": 14,
    "19432": 15,
    "19433": 16,
    "19480": 17,
    "19481": 18,
    "19524": 19,
    "19525": 20,
}

def process_derivative(input_file, output_dir):
    # Load the NIfTI image
    img = load(input_file)
    data = img.get_fdata()

    # Create masks based on the specified conditions
    mask_3 = data == 3
    mask_5 = data == 5
    mask_9 = data >= 9

    # Create new images for each mask
    img_3 = Nifti1Image(mask_3.astype(np.uint8), img.affine)
    img_5 = Nifti1Image(mask_5.astype(np.uint8), img.affine)
    img_9 = Nifti1Image(data * mask_9.astype(np.uint8), img.affine)

    # Save the new images
    basename = os.path.splitext(os.path.basename(input_file))[0]
    basename = os.path.splitext(os.path.basename(basename))[0].split("-")[-1]
    number = basename.split("_")[0]

    output_path_3 = os.path.join(output_dir, f"sub-{study_to_sub[number]}-{basename}_pmj.nii.gz")
    output_path_5 = os.path.join(output_dir, f"sub-{study_to_sub[number]}-{basename}_rootlet-bin.nii.gz")
    output_path_9 = os.path.join(output_dir, f"sub-{study_to_sub[number]}-{basename}_vertebrae.nii.gz")

    img_3.to_filename(output_path_3)
    img_5.to_filename(output_path_5)
    img_9.to_filename(output_path_9)

def process_subdirectories(root_dir, output_dir):
    for subdir, _, files in os.walk(root_dir):
        print(files)
        if len(files) > 0:
            if files[0].startswith("sub-"):
                for filename in files:
                    print(filename)
                    if filename.endswith(".nii.gz"):
                        input_file = os.path.join(subdir, filename)
                        process_derivative(input_file, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Process BIDS derivative files.")
    parser.add_argument("--input_dir", help="Path to the root directory of the BIDS dataset")
    parser.add_argument("--output_dir", help="Path to the output directory for processed files")
    args = parser.parse_args()

    process_subdirectories(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
