#!/usr/bin/env python
#
# This script is used to generate a figure showing the inter-session variability for individual sessions from the
# spine-generic single subject dataset.
# The script also computes a distance from the PMJ to the middle of each spinal level and COV. Results are saved to
# a CSV file.
#
# Usage:
#     python 01b_spine-generic_single-subject-generate_figure_inter-subject_variablity-PMJ_COV.py
#       -i /path/to/data_processed
#       -participants-tsv /path/to/participants.tsv
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

# color to assign to each MRI model for the figure
# Corresponds to https://github.com/spine-generic/spine-generic/blob/master/spinegeneric/cli/generate_figure.py#L112C1-L117C2
vendor_to_color = {
    "GE": "black",
    "Philips": "dodgerblue",
    "Siemens": "limegreen",
}

# Sort sites to match https://www.nature.com/articles/s41597-021-00941-8
list_of_sites = ['chiba750', 'perform', 'juntendo750w', 'tokyo750w', 'tokyoSigna1', 'tokyoSigna2',
                 'juntendoAchieva', 'ucl', 'chibaIngenia', 'glen', 'tokyoIngenia',
                 'juntendoPrisma', 'oxfordFmrib', 'unf', 'juntendoSkyra', 'poly', 'tokyoSkyra', 'douglas', 'mgh']

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
    parser.add_argument(
        '-participants-tsv',
        required=True,
        type=str,
        help='Path to the participants.tsv file from the spine-generic single subject dataset.'
    )

    return parser


def generate_figure(df, dir_path):
    """
    Generate a figure showing the inter-session variability for individual sessions and spinal levels.
    :param df: Pandas DataFrame with the data
    :param dir_path: Path to the data_processed folder
    :return: None
    """

    # Compute the mean distance from PMJ for each subject and spinal level.
    # Compute the coefficient of variation (COV) across subject for each spinal level. Also compute mean COV across
    # subjects for each manufacturer.
    cov_mean, cov_std = compute_mean_and_COV(df, dir_path)

    mpl.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()

    # rectangle width
    width = 0.475       # 0.1/4 * 19

    # Loop across subjects/sites
    for x, subject in enumerate(list_of_sites, 1):
        # Loop across spinal levels
        for level in df['spinal_level'].unique():
            row = df[(df['subject'] == subject) & (df['spinal_level'] == level)]

            # Skip if the row is empty
            if row.empty:
                continue

            # Get the distance from PMJ and height of spinal level
            start = float(row['distance_from_pmj_end'])
            height = float(row['height'])

            # Plot rectangle for the subject and spinal level
            ax.add_patch(
                patches.Rectangle(
                    xy=(x-width/2, start),
                    width=width,
                    height=height,
                    alpha=0.5,
                    color=vendor_to_color[row['manufacturer'].values[0]]
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
    ax.set_xlim(0.5, len(df['subject'].unique())+0.5)
    ax.set_ylim(min(df['distance_from_pmj_end'].min(), df['distance_from_pmj_start'].min())*0.9,
                max(df['distance_from_pmj_end'].max(), df['distance_from_pmj_start'].max())*1.05)

    # Set axis labels
    ax.set_xlabel('Site', fontsize=FONT_SIZE)
    ax.set_ylabel('Distance from PMJ [mm]', fontsize=FONT_SIZE)

    # Set subject name as x-axis ticks
    ax.set_xticks(range(1, len(list_of_sites)+1))
    ax.set_xticklabels(list_of_sites, fontsize=FONT_SIZE-4, rotation=30, ha='right')

    # Set size of y-axis ticks
    ax.tick_params(axis='y', labelsize=FONT_SIZE-4)

    # Put manufacturer name on top (centered in the middle site of each vendor), below the top title
    vendor_middle = {
        'GE': 3.5,
        'Philips': 9,
        'Siemens': 15.5,
        }
    for vendor in df['manufacturer'].unique():
        # Get the y coordinate of the top title
        y = ax.get_ylim()[0] - 3
        ax.text(
            vendor_middle[vendor], y,
            f"{vendor}\n(mean COV: {cov_mean['COV_'+vendor]:.2f} ± {cov_std['COV_'+vendor]:.2f}%)",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=FONT_SIZE-4,
            color=vendor_to_color[vendor],
            path_effects=[pe.withStroke(linewidth=0.5, foreground='black')]
        )

    # Set y-axis ticks to every 10 mm
    ax.set_yticks(range(40, 155, 10))

    # Reverse ylim
    ax.set_ylim(ax.get_ylim()[::-1])

    # Add horizontal grid
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    # Add title
    ax.set_title('Spinal Level Inter-Site Variability (spine-generic single-subject)', y=1.08, fontsize=FONT_SIZE)

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # Add bold red cross to 'sub-tokyoSigna1', 'sub-tokyoSigna2' sites to indicate that these sites were excluded
    # from COV computation
    ax.text(5, 38, 'x', fontsize=10, fontweight='bold', color='red', horizontalalignment='center')
    ax.text(6, 38, 'x', fontsize=10, fontweight='bold', color='red', horizontalalignment='center')

    plt.tight_layout()
    # Save the figure
    fname_figure = 'figure_inter_session_variability-spine-generic_single-subject.png'
    fig.savefig(os.path.join(dir_path, fname_figure), dpi=300)
    print(f'Figure saved to {os.path.join(dir_path, fname_figure)}')


def compute_mean_and_COV(df, dir_path):
    """
    Compute the mean distance from PMJ for each subject and spinal level.
    Compute the coefficient of variation (COV) across subjects, for each spinal level.
    Also compute mean COV across manufacturers.
    Create a table with the results and save it to a CSV file.
    :param df: Pandas DataFrame with the data
    :param dir_path: Path to the data_processed folder
    :return: cov_mean, cov_std: mean and standard deviation of the COV across sites for each manufacturer
    """

    results = []

    # Exclude sub-tokyoSigna1 and sub-tokyoSigna2 subjects
    df = df[~df['participant_id'].isin(['sub-tokyoSigna1', 'sub-tokyoSigna2'])]

    # Loop across subjects
    for subject in df['subject'].unique():
        # Loop across spinal levels
        for level in df['spinal_level'].unique():
            row = df[(df['subject'] == subject) & (df['spinal_level'] == level)]

            # Check if the row is empty
            if row.empty:
                results.append({'subject': subject, 'spinal_level': level, 'mean': 'n/a'})
            else:
                # Get the distance from PMJ and height of spinal level
                start = float(row['distance_from_pmj_end'])
                height = float(row['height'])

                mean = start + height / 2
                results.append({'subject': subject, 'spinal_level': level, 'mean': mean})

    # Create a pandas DataFrame from the parsed data
    df_results = pd.DataFrame(results)

    # Reformat the DataFrame to have spinal_levels as rows and subjects as columns
    df_results = df_results.pivot(index='spinal_level', columns='subject', values='mean')

    # Compute row-wise coefficient of variation (COV across subjects for each spinal level)
    df_results['COV'] = df_results.std(axis=1) / df_results.mean(axis=1) * 100

    # Compute mean COV across sites for each manufacturer
    for vendor in df['manufacturer'].unique():
        df_results[f'COV_{vendor}'] = df_results[[col for col in df_results.columns if vendor in df[df['subject'] == col]['manufacturer'].unique()]].std(axis=1) / \
                                      df_results[[col for col in df_results.columns if vendor in df[df['subject'] == col]['manufacturer'].unique()]].mean(axis=1) * 100

    # Compute mean COV across levels for each manufacturer
    cov_mean = df_results[['COV_GE', 'COV_Philips', 'COV_Siemens']].mean(axis=0)
    cov_std = df_results[['COV_GE', 'COV_Philips', 'COV_Siemens']].std(axis=0)

    # Save the DataFrame to a CSV file
    fname_csv = 'table_inter_session_variability-spine-generic_single-subject.csv'
    df_results.to_csv(os.path.join(dir_path, fname_csv))
    print(f'Table saved to {os.path.join(dir_path, fname_csv)}')

    return cov_mean, cov_std


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

    # Load participants.tsv file, load only participant_id and manufacturer columns
    participants_df = pd.read_csv(args.participants_tsv, sep='\t', usecols=['participant_id', 'manufacturer'])

    # Keep only cervical levels (2 to 8)
    df = df[df['spinal_level'].isin([2, 3, 4, 5, 6, 7, 8])]

    # Extract subjectID from the fname and add it as a column
    df['subject'] = df['fname'].apply(lambda x: x.split('_')[0])

    # Merge the two dataframes
    df = pd.merge(df, participants_df, left_on='subject', right_on='participant_id')

    # Sort the df based on manufacturer (GE, Philips, Siemens)
    df = df.sort_values(by=['manufacturer', 'subject', 'spinal_level'])

    # Remove "sub-" prefix from subject names
    df['subject'] = df['subject'].apply(lambda x: x.replace('sub-', ''))

    # Generate the figure
    generate_figure(df, dir_path)


if __name__ == '__main__':
    main()
