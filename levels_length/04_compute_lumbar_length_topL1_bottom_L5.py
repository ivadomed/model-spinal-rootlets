"""
This script computes the length from the top of L1 to the bottom of L5 from labeled spinal cord segmentations.

The top of L1 is its most superior slice and the bottom of L5 is its most inferior slice. The length is then the number
of slices from the top of L1 to the bottom of L5 (both included) multiplied by the slice thickness.

TODO: include also the centerline-based length to account for spinal cord curvature

Input:
- folder with manually labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz) with one value per
spinal level: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=S1, 7=S2.

Output (saved to the input folder):
    - lumbar_length_L1-L5.csv: length from the top of L1 to the bottom of L5 for each subject, and the min slice and
      max slice of each spinal level
    - figure_lumbar_length_L1-L5.png: the same length plotted per subject

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python 04_compute_lumbar_length_topL1_bottom_L5.py -i spinal_level_segmentation_L1-S2

Author: Jan Valosek
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spinalcordtoolbox.image import Image

LEVELS = {1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5', 6: 'S1', 7: 'S2'}
FIRST_LEVEL = 1  # L1
LAST_LEVEL = 5   # L5

FONT_SIZE = 14


def compute_length(fname):
    """
    Compute the length (in mm) from the top of FIRST_LEVEL to the bottom of LAST_LEVEL for one segmentation.
    :param fname: path to the labeled spinal cord segmentation
    :return: dict with the length in mm (NaN if one of the levels is missing) and, for each level, its min slice and
    max slice (NaN if the level is missing)
    """
    # Load the image and reorient it to RPI, so that the third axis is the S-I axis (slice 0 is the most inferior)
    img = Image(fname).change_orientation('RPI')
    data = img.data
    slice_thickness = img.dim[6]  # pz, slice thickness in mm

    # Min (most inferior) and max (most superior) slice of each spinal level
    row = {f'length_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm': np.nan}
    for value, name in LEVELS.items():
        z = np.where(data == value)[2]  # S-I index of each voxel of this level
        if len(z) == 0:
            print(f'  {name}: not found')
            row.update({f'{name}_slice_min': np.nan, f'{name}_slice_max': np.nan})
            continue
        row.update({f'{name}_slice_min': z.min(), f'{name}_slice_max': z.max()})
        print(f'  {name}: slices {z.min()}-{z.max()}')

    top = row[f'{LEVELS[FIRST_LEVEL]}_slice_max']      # top of L1
    bottom = row[f'{LEVELS[LAST_LEVEL]}_slice_min']    # bottom of L5
    if np.isnan(top) or np.isnan(bottom):
        print(f'  WARNING: {LEVELS[FIRST_LEVEL]} or {LEVELS[LAST_LEVEL]} missing; the length will be NaN.')
        return row

    # Length = number of slices from the top of L1 to the bottom of L5 (both included) x slice thickness
    n_slices = top - bottom + 1
    row[f'length_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm'] = n_slices * slice_thickness
    print(f'  Length top {LEVELS[FIRST_LEVEL]} - bottom {LEVELS[LAST_LEVEL]}: {n_slices} slices x '
          f'{slice_thickness:.2f} mm = {n_slices * slice_thickness:.1f} mm')

    return row


def generate_figure(df, dir_path):
    """
    Generate a figure with subjects on the x-axis and the length on the y-axis.
    :param df: dataframe with 'subject' and length columns
    :param dir_path: path to the output folder where the figure will be saved
    """
    x = range(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 5))
    # Convert mm to cm for plotting
    ax.scatter(x, df[f'length_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm'] / 10, color='red', s=50, zorder=3)

    # Subject names as x-axis labels
    ax.set_xticks(list(x))
    ax.set_xticklabels(df['subject'], rotation=45, ha='right')
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(0, 12)

    ax.set_ylabel(f'Length top {LEVELS[FIRST_LEVEL]} – bottom {LEVELS[LAST_LEVEL]} [cm]', fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)
    ax.tick_params(axis='x', labelsize=FONT_SIZE - 2)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    fname_figure = os.path.join(dir_path, f'figure_lumbar_length_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}.png')
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compute the length from the top of L1 to the bottom of L5.')
    parser.add_argument('-i', required=True,
                        help='Folder with labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz).')
    args = parser.parse_args()
    dir_path = os.path.abspath(args.i)

    # Compute the length for each subject
    rows = []
    for fname in sorted(glob.glob(os.path.join(dir_path, '**', '*_spinallevels_dseg.nii.gz'), recursive=True)):
        subject = os.path.basename(fname).split('_')[0]
        print(subject)
        rows.append({'subject': subject, **compute_length(fname)})
    df = pd.DataFrame(rows)

    # Save the lengths to a CSV file
    fname_csv = os.path.join(dir_path, f'lumbar_length_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}.csv')
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Plot the length per subject
    generate_figure(df, dir_path)


if __name__ == '__main__':
    main()
