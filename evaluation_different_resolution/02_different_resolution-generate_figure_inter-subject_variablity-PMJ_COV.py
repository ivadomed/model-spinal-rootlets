#!/usr/bin/env python
#
# This script is used to generate a figure showing the inter-resolution variability for on a single subject
# (sub-010_ses-headUp) from the Spinal Cord Head Position MRI dataset
# (https://openneuro.org/datasets/ds004507/versions/1.0.1).
#
# The script also computes a distance from the PMJ to the middle of each spinal level and COV. Results are saved to
# a CSV file.
#
# Usage:
#     python 02_different_resolution-generate_figure_inter-subject_variablity-PMJ_COV.py
#       -i /path/to/data_processed
#
# Authors: Jan Valosek
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


FONT_SIZE = 14


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Generate a figure showing the inter-session variability for individual sessions and spinal levels.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to the data_processed folder with CSV files for individual sessions generated by the '
             'inter-rater_variability/02a_rootlets_to_spinal_levels.py script.'
    )

    return parser


def generate_figure(df, dir_path):
    """
    Generate a figure showing the inter-session variability for individual sessions and spinal levels.
    :param df: Pandas DataFrame with the data
    :param dir_path: Path to the data_processed folder
    :return: None
    """
    mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()

    # rectangle width
    width = 0.25        # 0.1/4 * 10

    # Loop across sessions
    for x, session in enumerate(df['session'].unique(), 1):
        # Loop across spinal levels
        for level in df['spinal_level'].unique():
            row = df[(df['session'] == session) & (df['spinal_level'] == level)]

            # Skip if the row is empty
            if row.empty:
                continue

            # Get the distance from PMJ and height of spinal level
            start = float(row['distance_from_pmj_end'])
            height = float(row['height'])

            # Plot rectangle for the session and spinal level
            ax.add_patch(
                patches.Rectangle(
                    xy=(x-width/2, start),
                    width=width,
                    height=height,
                    alpha=0.5,
                    color='limegreen'
                ))

            # Add level number into each rectangle
            ax.text(
                x,     # x
                start,      # y
                int(level),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                color='white',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')]
            )

            # Add mean value
            ax.plot(
                [x-width/2, x+width/2],
                [start+height/2, start+height/2],
                color='black',
                linewidth=1,
                alpha=0.5,
                linestyle='dashed'
            )

    # Adjust the axis limits
    ax.set_xlim(0.5, len(df['session'].unique())+0.5)
    ax.set_ylim(min(df['distance_from_pmj_end'].min(), df['distance_from_pmj_start'].min())*0.9,
                max(df['distance_from_pmj_end'].max(), df['distance_from_pmj_start'].max())*1.05)

    # Set axis labels
    ax.set_xlabel('Resolution', fontsize=FONT_SIZE)
    ax.set_ylabel('Distance from PMJ [mm]', fontsize=FONT_SIZE)

    # Set session name as x-axis ticks
    ax.set_xticks(range(1, len(df['session'].unique())+1))
    ax.set_xticklabels(['0.6mm (orig)', '0.8mm', '1.0mm', '1.2mm', '1.4mm', '1.6mm'],
                       fontsize=FONT_SIZE-4, rotation=0, ha='center')

    # Set size of y-axis ticks
    ax.tick_params(axis='y', labelsize=FONT_SIZE-4)

    # Set y-axis ticks to every 10 mm
    ax.set_yticks(range(40, 155, 10))

    # Reverse ylim
    ax.set_ylim(ax.get_ylim()[::-1])

    # Add horizontal grid
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    # Add title
    ax.set_title('Spinal Level Inter-Resolution Variability (ds004507 sub-010_ses-headUp)', y=1.03, fontsize=FONT_SIZE)

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    # Save the figure
    fname_figure = 'figure_inter_resolution_variability-sub-010_ses-headUp.png'
    fig.savefig(os.path.join(dir_path, fname_figure), dpi=300)
    print(f'Figure saved to {os.path.join(dir_path, fname_figure)}')


def compute_mean_and_COV(df, dir_path):
    """
    Compute the mean distance from PMJ for each session and spinal level.
    Compute the coefficient of variation (COV) across sessions, for each spinal level.
    Also compute mean COV across manufacturers.
    Create a table with the results and save it to a CSV file.
    :param df: Pandas DataFrame with the data
    :param dir_path: Path to the data_processed folder
    :return: None
    """

    results = []

    # Loop across sessions
    for session in df['session'].unique():
        # Loop across spinal levels
        for level in df['spinal_level'].unique():
            row = df[(df['session'] == session) & (df['spinal_level'] == level)]

            # Check if the row is empty
            if row.empty:
                results.append({'session': session, 'spinal_level': level, 'mean': 'n/a'})
            else:
                # Get the distance from PMJ and height of spinal level
                start = float(row['distance_from_pmj_end'])
                height = float(row['height'])

                mean = start + height / 2
                results.append({'session': session, 'spinal_level': level, 'mean': mean})

    # Create a pandas DataFrame from the parsed data
    df_results = pd.DataFrame(results)

    # Reformat the DataFrame to have spinal_levels as rows and sessions columns
    df_results = df_results.pivot(index='spinal_level', columns='session', values='mean')

    # Compute row-wise coefficient of variation (COV across sessions for each spinal level)
    df_results['COV'] = df_results.std(axis=1) / df_results.mean(axis=1) * 100

    # Save the DataFrame to a CSV file
    fname_csv = 'table_inter_session_variability-courtois-neuromod.csv'
    df_results.to_csv(os.path.join(dir_path, fname_csv))
    print(f'Table saved to {os.path.join(dir_path, fname_csv)}')


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    dir_path = os.path.abspath(args.i)

    if not os.path.isdir(dir_path):
        print(f'ERROR: {args.i} does not exist.')
        sys.exit(1)

    # Get all the CSV files in the directory generated by the 02a_rootlets_to_spinal_levels.py script
    csv_files = glob.glob(os.path.join(dir_path, '**', '*label-rootlet*_pmj_distance.csv'), recursive=True)
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

    # Keep only cervical levels (2 to 8)
    df = df[df['spinal_level'].isin([2, 3, 4, 5, 6, 7, 8])]

    # Extract subjectID from the fname and add it as a column
    df['subject'] = df['fname'].apply(lambda x: x.split('_')[0])
    # Extract sessionID from the fname and add it as a column
    df['session'] = df['fname'].apply(lambda x: x.split('_')[1])

    # Sort the df based on session
    df = df.sort_values(by=['session'])

    # Generate the figure
    generate_figure(df, dir_path)

    # Compute the mean distance from PMJ for each subject and spinal level.
    # Compute the coefficient of variation (COV) across subject for each spinal level. Also compute mean COV across
    # manufacturers.
    compute_mean_and_COV(df, dir_path)


if __name__ == '__main__':
    main()
