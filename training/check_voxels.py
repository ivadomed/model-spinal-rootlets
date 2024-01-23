"""
This script opens a NIfTI image, finds the coordinates of voxels with a specified value,
and prints the voxel coordinates along with the maximum and minimum values in the image.

This is useful when recoding binary masks to a different value (e.g. 1 to 2) and you want
to check if the recoding was successful.

Usage:
python script.py <nii_image_path>

Jan Valosek
"""

import nibabel as nib
import numpy as np

def find_coordinates_with_value(image_path, target_value=1):
    # Load NIfTI image
    img = nib.load(image_path)

    # Get image data as a NumPy array
    data = img.get_fdata()

    # Find voxel coordinates with the specified value
    coordinates = np.argwhere(data == target_value)

    # Get max and min values in the image
    max_value = np.max(data)
    min_value = np.min(data)

    return coordinates, max_value, min_value


if __name__ == "__main__":
    import sys

    # Check if the input argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <nii_image_path>")
        sys.exit(1)

    # Get the NIfTI image path from the command line arguments
    nii_image_path = sys.argv[1]

    # Find coordinates with value 1 and max/min values in the NIfTI image
    voxel_coordinates, max_value, min_value = find_coordinates_with_value(nii_image_path)

    # Print the result
    print(f"Voxel coordinates with value 1 in {nii_image_path}:\n{voxel_coordinates}")
    print(f"Max value in the image: {max_value}")
    print(f"Min value in the image: {min_value}")
