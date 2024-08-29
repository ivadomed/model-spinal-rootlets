"""
This script is used to generate figures showing statistics of the overlap between rootlets and vertebrae levels for
individual levels.

The script reads CSV files generated by the inter-rater_variability/02a_rootlets_vertebrae_spinal_levels.py script or
pediatric_rootlets/discs_to_vertebral_levels.py script and calculates the overlap between rootlets and vertebrae levels
for each subject and spinal level. The overlap is calculated as the percentage of the shared area between the two levels
relative to the total area of the spinal level.
Usage: python analysis_comparison_spinal_vs_vertebral_levels.py -i /path/to/data_processed
-participants /path/to/participants.tsv -o /path/to/output_folder
"""

import os
import sys
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# Initialize dictionaries
SUBJECT_TO_AXIS = {}
SUBJECT_TO_XTICKS = {}
LIST_OF_LEVEL_TYPES = ['rootlets', 'vertebrae']
XOFFSET = {'rootlets': -0.10, 'vertebrae': 0.10}
LEVEL_TYPE_COLOR = {'rootlets': 'red', 'vertebrae': 'green'}
SPINAL_LEVEL_TYPES_TO_LEGEND = {'rootlets': 'spinal', 'vertebrae': 'vertebral'}
FONT_SIZE = 14

AGE_GROUPS = {
    '7-10': range(7, 11),
    '11-14': range(11, 15),
    '15-17': range(15, 18)
}


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Generate a figures showing statistics of the overlap between rootlets and vertebrae levels',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to the data_processed folder with CSV files for individual subjects and method of level estimation generated by the '
             'inter-rater_variability/02a_rootlets_vertebrae_spinal_levels.py script or pediatric_rootlets/discs_to_vertebral_levels.py.'
             'The figure will be saved to the same folder.'
    )
    parser.add_argument(
        '-participants',
        required=True,
        type=str,
        help='Path to the participants.tsv file. The file is used to fetch sex and age for individual subjects.'
    )
    parser.add_argument(
        '-o',
        required=True,
        type=str,
        help='Path to the output folder, where figures will be saved.'
    )

    return parser


def calculate_overlap(df):
    """
    Calculate overlap between rootlets and vertebrae levels for each subject and spinal level.
    :param df: Input dataframe with data for individual subjects and spinal levels.
    :return: Dataframe with overlap data for individual subjects and spinal levels.
    """

    # Initialize lists to store data
    list_subjects = []
    list_levels = []
    list_overlap= []
    list_sex = []
    list_age = []
    list_levels_num = []

    # get unique subjects:
    subjects = df['subject'].unique()

    # get sex of subjects:
    sex_dict = {subject: df[df['subject'] == subject]['sex'].iloc[0] for subject in subjects}
    age_dict = {subject: df[df['subject'] == subject]['age'].iloc[0] for subject in subjects}

    # get unique spinal levels:
    spinal_levels = df['spinal_level'].unique()

    # sort spinal levels as integers:
    spinal_levels = sorted(spinal_levels, key=lambda x: int(x))

    # Calculate overlap between rootlets and vertebrae for each subject and spinal level:
    # Calculate overlap between rootlets and vertebrae for each subject and spinal level:
    for subject in subjects:
        for spinal_level in spinal_levels:
            # NOTE: in the CSV file, the start and end are switched, so we need to switch them here:
            # Get start and end rootlets for the subject and spinal level:
            end_rootlets = df[(df['subject'] == subject) & (df['spinal_level'] == spinal_level + 1) & (
                        df['spinal_level_type'] == 'rootlets')]['distance_from_pmj_start'].values
            start_rootlets = df[(df['subject'] == subject) & (df['spinal_level'] == spinal_level + 1) & (
                        df['spinal_level_type'] == 'rootlets')]['distance_from_pmj_end'].values

            # Get start and end vertebrae for the subject and spinal level:
            end_vertebrae = df[(df['subject'] == subject) & (df['spinal_level'] == spinal_level) & (
                        df['spinal_level_type'] == 'vertebrae')]['distance_from_pmj_start'].values
            start_vertebrae = df[(df['subject'] == subject) & (df['spinal_level'] == spinal_level) & (
                        df['spinal_level_type'] == 'vertebrae')]['distance_from_pmj_end'].values

            # If there is no data for the subject and spinal level, skip:
            if not end_vertebrae or not start_vertebrae or not end_rootlets or not start_rootlets:
                break
            else:
                end_rootlets = end_rootlets[0]
                start_rootlets = start_rootlets[0]
                end_vertebrae = end_vertebrae[0]
                start_vertebrae = start_vertebrae[0]

                # Calculate the overlap directly using max and min functions:
                overlap_start = max(start_rootlets, start_vertebrae)
                overlap_end = min(end_rootlets, end_vertebrae)

                # Check if there is an overlap:
                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    total_rootlets_length = end_rootlets - start_rootlets
                    overlap = (overlap_length / total_rootlets_length) * 100
                else:
                    overlap = 0

                # append data to lists:
                list_subjects.append(subject)
                list_levels.append(f"C{spinal_level}-C{spinal_level+1}")
                list_levels_num.append(spinal_level-2)
                list_overlap.append(overlap)
                list_sex.append(sex_dict[subject])
                list_age.append(age_dict[subject])

    # save overlap data to dataframe:
    df_overlap = pd.DataFrame({'subject': list_subjects, 'spinal_level': list_levels_num, 'spinal_level_name': list_levels, 'overlap': list_overlap, 'sex': list_sex, 'age': list_age})

    return df_overlap


def create_boxplot(df_overlap, output_path):
    # create boxplot with seaborn (x-ax is spinal level, y-ax is overlap):
    plt.figure()
    sns.set_style("darkgrid")

    # Create the boxplot with hue based on sex
    ax = sns.boxplot(x='spinal_level', y='overlap', data=df_overlap, hue='sex', boxprops=dict(alpha=.5), legend=False)
    colors = [["#4374B3"], ["darkorange"]]
    idx = 0
    for sex in df_overlap['sex'].unique():
        if sex == 'M':
            df_overlap['spinal_level'] = df_overlap['spinal_level'] - 0.2
        else:
            df_overlap['spinal_level'] = df_overlap['spinal_level'] + 0.4

        sns.set_palette(sns.color_palette(colors[idx]))
        sns.scatterplot(x='spinal_level', y='overlap', size='age', data=df_overlap[df_overlap['sex'] == sex], sizes=(10, 150), hue='sex',
                          edgecolor='w', legend=False)
        sns.set_style("darkgrid")
        idx += 1

    ax.set_title('Overlap between rootlets and vertebrae by Sex and Age')
    ax.set_ylabel('Overlap [%]')
    ax.set_xlabel('Spinal level')
    ax.set_xticklabels(df_overlap['spinal_level_name'].unique())
    ax.set_yticks(range(0, 101, 10))

    lines = [
        Line2D([0], [0], color='#4374b3', linewidth=3, linestyle='-', label='Male'),
        Line2D([0], [0], color='#ff8c00', linewidth=3, linestyle='-', label='Female')
    ]
    age_circles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Lower Age'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Higher Age')
    ]
    handles = lines + age_circles

    # Create the legend
    plt.legend(handles=handles, loc='lower left', ncol=2, fontsize=10, handlelength=1)

    plt.savefig(f'{output_path}/boxplot_overlap_rootlets_vertebrae.svg', dpi=300)
    plt.show()


def create_scatterplot(df_overlap, output_path):
    sns.set_palette('muted')
    g = sns.FacetGrid(df_overlap, col='spinal_level_name', hue='sex', col_wrap=3, height=4, ylim=(0,100))
    g.map(sns.scatterplot, 'age', 'overlap')
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Scatterplots of Overlap by Age, Sex, and Spinal Level')
    plt.savefig(f'{output_path}/scatterplot_overlap_rootlets_vertebrae.svg', dpi=300)
    plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()

    dir_path = os.path.abspath(args.i)
    output_path = os.path.abspath(args.o)

    if not os.path.isdir(dir_path):
        print(f'ERROR: {args.i} does not exist.')
        sys.exit(1)

    df_participants = pd.read_csv(args.participants, sep='\t')
    participants_age = df_participants[['participant_id', 'age']]
    participants_sex = df_participants[['participant_id', 'sex']]

    # Get all the CSV files in the directory generated by the 02a_rootlets_to_spinal_levels.py script
    csv_files = glob.glob(os.path.join(dir_path, '**', '*pmj_distance_*[vertebral_disc|rootlets].csv'), recursive=True)

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
    df['age'] = df['fname'].apply(lambda x: participants_age[participants_age['participant_id'] == x[:7]]['age'].values[0])
    df['age_group'] = df['age'].apply(
        lambda x: '7-10' if x in AGE_GROUPS['7-10'] else '11-14' if x in AGE_GROUPS['11-14'] else '15-17')

    # Extract rootlets or vertebrae level type from the fname and add it as a column
    df['spinal_level_type'] = df['fname'].apply(
        lambda x: 'rootlets' if 'label-rootlets' in x else 'vertebrae'
    )

    # Extract subjectID from the fname and add it as a column
    df['subject'] = df['fname'].apply(lambda x: x.split('_')[0])

    # Keep spinal levels C2-C8) and vertebral levels C2-C7
    df = df[((df['spinal_level_type'] == 'rootlets') & (df['spinal_level'].isin([2, 3, 4, 5, 6, 7, 8]))) |
            ((df['spinal_level_type'] == 'vertebrae') & (df['spinal_level'].isin([2, 3, 4, 5, 6, 7])))]

    # Define the custom order
    age_group_order = ['7-10', '11-14', '15-17']

    # Convert the 'age_group' column to a categorical type with the custom order
    df['age_group'] = pd.Categorical(df['age_group'], categories=age_group_order, ordered=True)

    df['sex'] = df['fname'].apply(
        lambda x: participants_sex[participants_sex['participant_id'] == x[:7]]['sex'].values[0])

    # Sort the DataFrame based on the 'age_group' column
    df = df.sort_values('age').reset_index(drop=True)

    # Calculate overlap between rootlets and vertebrae levels for each subject and spinal level
    overlap_table = calculate_overlap(df)

    # Create boxplot and scatterplot
    create_boxplot(overlap_table, output_path)
    create_scatterplot(overlap_table, output_path)


if __name__ == '__main__':
    main()