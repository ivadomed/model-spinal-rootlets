#!/usr/bin/env python
#
# This script is used to generate a figure showing the variability between spinal levels and vertebrae levels.
# The script also computes a distance from the PMJ to the middle of each spinal level and COV. Results are saved to
# a CSV file.
#
# Usage:
#     python generate_figure_rootlets_vertebrae_spinal_levels.py -i /path/to/data_processed
#
# Authors: Katerina Krejci, Jan Valosek
#

import os
import sys
import glob
import argparse

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

from scipy.stats import mannwhitneyu

SUBJECT_TO_AXIS = {
    'sub-101': 1,
    'sub-102': 2,
    'sub-103': 3,
    'sub-104': 4,
    'sub-105': 5,
}
SUBJECT_TO_XTICKS = {
    'sub-101': 'sub-101',
    'sub-102': 'sub-102',
    'sub-103': 'sub-103',
    'sub-104': 'sub-104',
    'sub-105': 'sub-105',
}
LIST_OF_LEVEL_TYPES = ['rootlets', 'vertebrae']
XOFFSET = {'rootlets': -0.10, 'vertebrae': 0.10}
LEVEL_TYPE_COLOR = {'rootlets': 'red', 'vertebrae': 'green'}
SPINAL_LEVEL_TYPES_TO_LEGEND = {'rootlets': 'spinal', 'vertebrae': 'vertebral'}
FONT_SIZE = 14


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Generate a figure showing the variability of spinal and vertebral levels for individual subjects and spinal levels.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to the data_processed folder with CSV files for individual subjects and method of level estimation generated by the '
             '02a_rootlets_vertebrae_spinal_levels.py script.'
             'The figure will be saved to the same folder.'
    )

    return parser


def generate_figure(df, dir_path):
    """
    Generate a figure showing the variability of spinal and vertebral levels for individual subjects and spinal levels.
    :param df: Pandas DataFrame with the data
    :param dir_path: Path to the data_processed folder
    :return: None
    """
    #mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot()

    # Loop across subjects
    for subject in SUBJECT_TO_AXIS.keys():
        # Loop across spinal level types
        for spinal_level_type in LIST_OF_LEVEL_TYPES:
            # Loop across spinal levels
            for level in df['spinal_level'].unique():
                row = df[(df['subject'] == subject) & (df['spinal_level_type'] == spinal_level_type) & (df['spinal_level'] == level)]

                # Skip if the row is empty
                if row.empty:
                    continue

                # Get the distance from PMJ and height of spinal level
                start = float(row['distance_from_pmj_end'])
                height = float(row['height'])

                # Plot rectangle for the subject, spinal and vertebral levels
                ax.add_patch(
                    patches.Rectangle(
                        (SUBJECT_TO_AXIS[subject] + XOFFSET[spinal_level_type], start),      # (x,y)
                        0.1,            # width
                        height,         # height
                        color=LEVEL_TYPE_COLOR[spinal_level_type],
                        alpha=0.5,
                    ))

                # Add level number into each rectangle
                ax.text(
                    SUBJECT_TO_AXIS[subject] + XOFFSET[spinal_level_type] + 0.05,     # x
                    start,                                                  # y
                    int(level),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    color='white',
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')]
                )

                # Add mean value
                ax.plot(
                    [SUBJECT_TO_AXIS[subject] + XOFFSET[spinal_level_type], SUBJECT_TO_AXIS[subject] + XOFFSET[spinal_level_type] + 0.1],
                    [start+height/2, start+height/2],
                    color='black',
                    linewidth=1,
                    alpha=0.5,
                    linestyle='dashed'
                )

    # Adjust the axis limits
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(min(df['distance_from_pmj_end'].min(), df['distance_from_pmj_start'].min())*0.9,
                max(df['distance_from_pmj_end'].max(), df['distance_from_pmj_start'].max())*1.05)

    # Set axis labels
    ax.set_xlabel('Subject', fontsize=FONT_SIZE)
    ax.set_ylabel('Distance from PMJ [mm]', fontsize=FONT_SIZE)

    # Set x-axis ticklabels based on the SUBJECT_TO_AXIS dictionary
    ax.set_xticks(list(SUBJECT_TO_AXIS.values()))
    ax.set_xticklabels(list(SUBJECT_TO_XTICKS.values()), fontsize=FONT_SIZE-4)

    # Set size of y-axis ticks
    ax.tick_params(axis='y', labelsize=FONT_SIZE-4)

    # Set y-axis ticks to every 10 mm
    ax.set_yticks(range(10, 160, 10))

    # Reverse ylim
    ax.set_ylim(ax.get_ylim()[::-1])

    # Add horizontal grid
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    # Add a single line legend with opacity 0.5
    ax.legend(handles=[
        patches.Patch(color=LEVEL_TYPE_COLOR[spinal_level_type], label=SPINAL_LEVEL_TYPES_TO_LEGEND[spinal_level_type], alpha=0.5)
        for spinal_level_type in LIST_OF_LEVEL_TYPES
    ], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.12))        # bbox_to_anchor=(0.5, -.1)
    # Update the legend title font size
    plt.setp(ax.get_legend().get_title(), fontsize=FONT_SIZE-2)

    # Add title and move it slightly up
    ax.set_title('Spinal vs. Vertebral Levels', pad=40)        # pad=20

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    plt.tight_layout()

    # Save the figure
    fname_figure = 'figure_spinal_levels_vs_vertebral_levels.png'
    fig.savefig(os.path.join(dir_path, fname_figure), dpi=300)
    print(f'Figure saved to {os.path.join(dir_path, fname_figure)}')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    dir_path = os.path.abspath(args.i)

    if not os.path.isdir(dir_path):
        print(f'ERROR: {args.i} does not exist.')
        sys.exit(1)

    # Get all the CSV files in the directory generated by the 02a_rootlets_to_spinal_levels.py script
    csv_files = glob.glob(os.path.join(dir_path, '**', '*pmj_distance_*[vertebral|rootlets].csv'), recursive=True)

    # if csv_files is empty, exit
    if len(csv_files) == 0:
        print(f'ERROR: No CSV files found in {dir_path}')

    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Loop across CSV files and aggregate the results into pandas dataframe
    for csv_file in csv_files:
        df_file = pd.read_csv(csv_file)
        parsed_data.append(df_file)

    # Combine list of dataframes into one dataframe
    df = pd.concat(parsed_data)

    # Extract rootlets or vertebrae level type from the fname and add it as a column
    df['spinal_level_type'] = df['fname'].apply(
        lambda x: 'rootlets' if x.split('_')[2].split('-')[1] == 'rootlets' else 'vertebrae'
    )

    # Extract subjectID from the fname and add it as a column
    df['subject'] = df['fname'].apply(lambda x: x.split('_')[0])

    # Extract spinal level (cervical 2-8) and vertebral level (1-7)
    df = df[((df['spinal_level_type'] == 'rootlets') & (df['spinal_level'].isin([2, 3, 4, 5, 6, 7, 8]))) | ((df['spinal_level_type'] == 'vertebrae') & (df['spinal_level'].isin([1, 2, 3, 4, 5, 6, 7])))]

    # Generate the figure
    generate_figure(df, dir_path)


if __name__ == '__main__':
    main()

