"""
Recode lumbar rootlets values to be consecutive starting from 0 (zero represents background).
The consecutive labels order is required by the nnUNet framework.

For example, the script can recode:
    [ 0. 20. 21. 22. 23. 24. 25. 26. 27.]
Into:
    [0. 1. 2. 3. 4. 5. 6. 7. 8.]

Example usage:
    python recode_nii.py -i /path/to/input.nii.gz -o /path/to/output.nii.gz

Author: Jan Valosek
"""

import nibabel as nib
import numpy as np
import argparse


def recode_values(nii_file_path, output_file_path):
    """
    Recode values in a NIfTI image to consecutive integers starting from 0 (zero represents background).

    Args:
    nii_file_path (str): Path to the input NIfTI file.
    output_file_path (str): Path to save the recoded NIfTI file.
    """

    print(f"Input NIfTI file: {nii_file_path}")

    # Load the NIfTI file
    nii = nib.load(nii_file_path)
    data = nii.get_fdata()

    # Get unique values in the data
    unique_values = np.unique(data)
    print(f"Unique values in the data: {unique_values}")

    # Create a mapping from original values to consecutive values
    value_mapping = {original: new for new, original in enumerate(unique_values)}

    # Recode the data using the mapping
    recoded_data = np.vectorize(value_mapping.get)(data)
    # Get unique values in the recoded data
    recoded_unique_values = np.unique(recoded_data)
    print(f"Unique values in the recoded data: {recoded_unique_values}")

    # Create a new NIfTI image with the recoded data
    recoded_nii = nib.Nifti1Image(recoded_data, nii.affine, nii.header)

    # Save the recoded NIfTI image
    nib.save(recoded_nii, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recode values in a NIfTI image to consecutive integers starting "
                                                 "from 0 (zero represents background).")
    parser.add_argument("-i", type=str, help="Path to the input NIfTI file.")
    parser.add_argument("-o", type=str, help="Path to save the recoded NIfTI file.")

    args = parser.parse_args()

    recode_values(args.i, args.o)
