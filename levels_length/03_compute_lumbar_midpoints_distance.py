"""
This script computes the distance between the L1 and S1 spinal level midpoints from labeled spinal cord segmentations.

The midpoint of each spinal level is the center of mass of the labeled spinal cord segmentation.
The total distance is then the number of slices between the L1 and S1 midpoints multiplied by the slice thickness.

TODO: include also the centerline-based distance to account for spinal cord curvature

Input:
- folder with manually labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz) with one value per
spinal level: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=S1, 7=S2.

Output (saved to the input folder):
    - lumbar_midpoints_distance_L1-S1.csv: distance between the L1 and S1 midpoints for each subject, and the min
      slice, max slice and midpoint slice of each spinal level
    - figure_lumbar_midpoints_distance_L1-S1.png: the same distance plotted per subject

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python 03_compute_lumbar_midpoints_distance.py -i spinal_level_segmentation_L1-S2

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
LAST_LEVEL = 6   # S1

FONT_SIZE = 14

# Excel file with the subject sex ('sex' column in the 'Spinal level heights (mm)' sheet)
PARTICIPANTS_XLSX = os.path.expanduser('~/results/lumbar_rootlets_Balgrist/spinal_level_segmentation_L1-S2/'
                                       'spinal_level_heights_top_L1-bottom_S2_with_chart.xlsx')
PARTICIPANTS_SHEET = 'Spinal level heights (mm)'

# Female and male colors and markers
SEX_STYLE = {'F': {'color': '#2a78d6', 'marker': 'o', 'label': 'Female'},
             'M': {'color': '#eb6834', 'marker': 's', 'label': 'Male'}}


def compute_midpoints_distance(fname):
    """
    Compute the distance (in mm) between the FIRST_LEVEL and LAST_LEVEL midpoints for one segmentation.
    :param fname: path to the labeled spinal cord segmentation
    :return: dict with the distance in mm (NaN if one of the levels is missing) and, for each level, its min slice,
    max slice and midpoint slice (NaN if the level is missing)
    """
    # Load the image and reorient it to RPI, so that the third axis is the S-I axis
    img = Image(fname).change_orientation('RPI')
    data = img.data
    slice_thickness = img.dim[6]  # pz, slice thickness in mm

    # Center of mass of each spinal level along the S-I axis (in slices)
    row = {f'midpoints_distance_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm': np.nan}
    midpoints = {}
    for value, name in LEVELS.items():
        z = np.where(data == value)[2]  # S-I index of each voxel of this level
        if len(z) == 0:
            print(f'  {name}: not found')
            row.update({f'{name}_slice_min': np.nan, f'{name}_slice_max': np.nan, f'{name}_midpoint_slice': np.nan})
            continue
        midpoints[value] = z.mean()
        row.update({f'{name}_slice_min': z.min(), f'{name}_slice_max': z.max(), f'{name}_midpoint_slice': z.mean()})
        print(f'  {name}: midpoint at slice {midpoints[value]:.1f} (slices {z.min()}-{z.max()})')

    if FIRST_LEVEL not in midpoints or LAST_LEVEL not in midpoints:
        print(f'  WARNING: {LEVELS[FIRST_LEVEL]} or {LEVELS[LAST_LEVEL]} missing; the distance will be NaN.')
        return row

    # Distance between the midpoints = number of slices x slice thickness
    n_slices = midpoints[FIRST_LEVEL] - midpoints[LAST_LEVEL]
    row[f'midpoints_distance_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm'] = n_slices * slice_thickness
    print(f'  Distance {LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]} midpoints: {n_slices:.1f} slices x '
          f'{slice_thickness:.2f} mm = {row[f"midpoints_distance_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm"]:.1f} mm')

    return row


def generate_figure(df, dir_path):
    """
    Generate a figure with subjects on the x-axis and the distance between the midpoints on the y-axis.
    :param df: dataframe with 'subject' and 'midpoints_distance_mm' columns
    :param dir_path: path to the output folder where the figure will be saved
    The dataframe must also contain a 'sex' column ('F'/'M') used to color the points.
    """
    df_plot = df

    fig, ax = plt.subplots(figsize=(max(6, len(df_plot) * 0.6), 5))
    # Points colored by sex; subjects without sex information in gray
    x = np.arange(1, len(df_plot) + 1)
    y = (df_plot[f'midpoints_distance_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}_mm']).to_numpy() / 10  # convert mm to cm for plotting
    for sex, style in SEX_STYLE.items():
        mask = (df_plot['sex'] == sex).to_numpy()
        ax.scatter(x[mask], y[mask], color=style['color'], marker=style['marker'], s=50, zorder=3,
                   label=f"{style['label']} (n = {mask.sum()})")
    mask = (~df_plot['sex'].isin(list(SEX_STYLE))).to_numpy()
    if mask.any():
        ax.scatter(x[mask], y[mask], color='gray', marker='x', s=50, zorder=3,
                   label=f'Sex unknown (n = {mask.sum()})')

    # Subject names as x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(df['subject'], rotation=45, ha='right')
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(0, 12)

    ax.set_ylabel(f'Distance {LEVELS[FIRST_LEVEL]}–{LEVELS[LAST_LEVEL]} midpoints [cm]', fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)
    ax.tick_params(axis='x', labelsize=FONT_SIZE - 2)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Legend above the plot so it does not overlap the points
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, fontsize=FONT_SIZE - 4,
              handletextpad=0.3, columnspacing=1)

    plt.tight_layout()

    fname_figure = os.path.join(dir_path,
                                f'figure_lumbar_midpoints_distance_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}.png')
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compute the distance between the L1 and S1 spinal level midpoints.')
    parser.add_argument('-i', required=True,
                        help='Folder with labeled spinal cord segmentations (sub-*/*_spinallevels_dseg.nii.gz).')
    parser.add_argument('-xlsx', default=PARTICIPANTS_XLSX,
                        help=f'Excel file with the subject sex (used to color the points). '
                             f'Default: {PARTICIPANTS_XLSX}')
    args = parser.parse_args()
    dir_path = os.path.abspath(args.i)

    # Compute the distance for each subject
    rows = []
    for fname in sorted(glob.glob(os.path.join(dir_path, '**', '*_spinallevels_dseg.nii.gz'), recursive=True)):
        subject = os.path.basename(fname).split('_')[0]
        print(subject)
        rows.append({'subject': subject, **compute_midpoints_distance(fname)})
    df = pd.DataFrame(rows)

    # Save the distances to a CSV file
    fname_csv = os.path.join(dir_path, f'lumbar_midpoints_distance_{LEVELS[FIRST_LEVEL]}-{LEVELS[LAST_LEVEL]}.csv')
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Add the subject sex (used only to color the points in the figure)
    df_sex = pd.read_excel(args.xlsx, sheet_name=PARTICIPANTS_SHEET)
    df_sex = df_sex[df_sex['Participants'].astype(str).str.startswith('sub-')]
    df_sex['sex'] = df_sex['sex'].str.upper()   # 'f'/'m' -> 'F'/'M'
    df = df.merge(df_sex[['Participants', 'sex']], how='left', left_on='subject', right_on='Participants')

    # Plot the distance per subject
    generate_figure(df, dir_path)


if __name__ == '__main__':
    main()
