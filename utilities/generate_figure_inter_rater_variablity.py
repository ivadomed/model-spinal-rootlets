#!/usr/bin/env python
#
# This script is used to generate a figure showing the inter-rater variability for individual subjects and spinal
# levels.
#
# Usage:
#     python generate_figure_inter_rater_variability.py -i /path/to/data_processed
#
# Authors: Jan Valosek
#

import os
import glob
import argparse

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SUBJECT_TO_AXIS = {
    'sub-barcelona01': 1,
    'sub-brnoUhb03': 2,
    'sub-amu02': 3,
    'sub-007': 4,
    'sub-010': 5,
}
SUBJECT_TO_XTICKS = {
    'sub-barcelona01': 'sub-barcelona01',
    'sub-brnoUhb03': 'sub-brnoUhb03',
    'sub-amu02': 'sub-amu02',
    'sub-007': 'sub-007_ses-headNormal',
    'sub-010': 'sub-010_ses-headUp',
}
LIST_OF_RATER = ['rater1', 'rater2', 'rater3']
RATER_XOFFSET = {'rater1': -0.2, 'rater2': -0.05, 'rater3': 0.1}
RATER_COLOR = {'rater1': 'red', 'rater2': 'green', 'rater3': 'blue'}


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Generate a figure showing the inter-rater variability for individual subjects and spinal levels.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to the data_processed folder with CSV files for individual subjects and raters.'
    )

    return parser


def generate_figure(df, dir_path):

    mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot()

    # Loop across subjects
    for subject in SUBJECT_TO_AXIS.keys():
        # Loop across raters
        for rater in LIST_OF_RATER:
            # Loop across spinal levels
            for level in df['spinal_level'].unique():
                row = df[(df['subject'] == subject) & (df['rater'] == rater) & (df['spinal_level'] == level)]
                # Get the distance from PMJ and height of spinal level
                start = float(row['distance_from_pmj_end'])
                height = float(row['height'])

                # Plot rectangle for the subject, rater and spinal level
                ax.add_patch(
                    patches.Rectangle(
                        (SUBJECT_TO_AXIS[subject]+RATER_XOFFSET[rater], start),      # (x,y)
                        0.1,            # width
                        height,         # height
                        color=RATER_COLOR[rater],
                        alpha=0.5,
                    ))

    # Adjust the axis limits
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(min(df['distance_from_pmj_end'].min(), df['distance_from_pmj_start'].min())*0.9,
                max(df['distance_from_pmj_end'].max(), df['distance_from_pmj_start'].max())*1.05)

    # Set axis labels
    ax.set_xlabel('Subject')
    ax.set_ylabel('Distance from PMJ [mm]')

    # Set x-axis ticklabels based on the SUBJECT_TO_AXIS dictionary
    ax.set_xticks(list(SUBJECT_TO_AXIS.values()))
    ax.set_xticklabels(list(SUBJECT_TO_XTICKS.values()))

    # Set y-axis ticks to every 10 mm
    ax.set_yticks(range(40, 160, 10))

    # Reverse ylim
    ax.set_ylim(ax.get_ylim()[::-1])

    # Add horizontal grid with transparency 0.2 every 10 mm
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    # Add legend with raters with opacity 0.5
    ax.legend(handles=[
        patches.Patch(color=RATER_COLOR[rater], label=rater, alpha=0.5)
        for rater in LIST_OF_RATER
    ])

    # Save the figure
    fname_figure = 'figure_inter_rater_variability.png'
    fig.savefig(os.path.join(dir_path, fname_figure), dpi=300)
    print(f'Figure saved to {os.path.join(dir_path, fname_figure)}')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    dir_path = os.path.abspath(args.i)

    if not os.path.isdir(dir_path):
        print(f'ERROR: {args.i} does not exist.')

    # Get all the CSV files in the directory
    csv_files = glob.glob(os.path.join(dir_path, '**', '*label-rootlet_rater*.csv'), recursive=True)
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

    # Keep only levels 2 to 8
    df = df[df['spinal_level'].isin([2, 3, 4, 5, 6, 7, 8])]

    # Extract rater from the fname and add it as a column
    df['rater'] = df['fname'].apply(lambda x: x.split('_')[-1].split('.')[0])
    # Extract subjectID from the fname and add it as a column
    df['subject'] = df['fname'].apply(lambda x: x.split('_')[0])

    # Generate the figure
    generate_figure(df, dir_path)


if __name__ == '__main__':
    main()
