"""
This script makes a visual comparison of the validation pseudo dice across epochs for multiple nnUNet training logs.
The script reads the log files, extracts the epoch number and validation pseudo dice, and plots them.

NOTE: This script is a modified version of
https://github.com/ivadomed/utilities/blob/main/training_scripts/plot_nnunet_training_log.py script, where the
functionality was extended to compare multiple training logs in specific spinal level.

Usage:
    python plot_nnunet_different_training_logs.py -i <path_to_log_file(s)> -spinal-level <spinal_level>
    -output-image <output_image>
"""

import os
import re
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Plot epoch number vs Pseudo dice based on a nnUNet training log file for multiple logs '
                    '- you can choose the spinal level for visualization.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        nargs='+',
        type=str,
        help='Path to the txt log file produced by nnUNet (or more log files separated by space).'
             'Example: training_log_2024_1_22_11_09_18.txt'
    )
    parser.add_argument(
        '-spinal-level',
        required=True,
        nargs='+',
        type=str,
        help='Spinal level of the data for comparison training logs.'
    )
    parser.add_argument(
        '-output-image',
        required=False,
        type=str,
        help='Path to the output image file, where you want to save the plot.'

    )

    return parser


def extract_epoch_and_dice(log_file_path):
    """
    Extract fold number and epoch and pseudo dice from the log file.
    Args:
        log_file_path: Path to the log file.
    Returns:
        epoch_and_dice_data: List of dictionaries with keys 'epoch' and 'pseudo_dice'.
        fold_number: Fold number used for training.
    """
    epoch_and_dice_data = []
    fold_pattern = re.compile(r'Desired fold for training: (\d+)')
    epoch_pattern = re.compile(r'Epoch (\d+)')
    dice_pattern = re.compile(r'Pseudo dice \[([^,\]]+(?:, [^,\]]+)*)\]')
    fold_number = None

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Extracting the fold number, epoch number and pseudo dice
    for line in lines:
        fold_match = fold_pattern.search(line)
        epoch_match = epoch_pattern.search(line)
        dice_match = dice_pattern.search(line)

        if fold_match:
            fold_number = int(fold_match.group(1))

        if epoch_match:
            epoch = int(epoch_match.group(1))
            epoch_and_dice_data.append({'epoch': epoch, 'pseudo_dice': None})

        elif dice_match:
            # Extracting the list using regular expression
            list_match = re.search(r'\[.*\]', dice_match.group())

            if list_match:
                extracted_list_str = list_match.group(0)
                # Replace 'nan' with the actual nan value
                extracted_list_str = extracted_list_str.replace('nan', 'float("nan")')
                # Evaluating the string representation of the list
                extracted_list = eval(extracted_list_str)
                epoch_and_dice_data[-1]['pseudo_dice'] = extracted_list

    # Check if fold_number is instance of int ("if not fold_number:" does not work for '0')
    if not isinstance(fold_number, int):
        fold_number = 'all'

    return epoch_and_dice_data, fold_number


def create_figure(df, fname_fig, spinal_level, contrasts):
    """
    Create a Seaborn line plot for the validation pseudo dice across epochs and different training logs.
    :param df: DataFrame containing the epoch number and validation pseudo dice for each class.
    :param fname_fig: Output figure file path.
    :param spinal_level: User-defined spinal level for visualization.
    :param contrasts: Contrast names for the plot title.
    :return:
    """

    # Melt the DataFrame for Seaborn plotting
    df_melted = df.melt(id_vars=['epoch'], var_name='class', value_name='pseudo_dice')

    # Create the Seaborn line plot
    plt.figure(figsize=(12, 6))
    sns.set(style="dark", palette="Paired")
    sns.set_theme()

    # check number of classes - for setting the color palette:
    num_classes = len(df_melted['class'].unique())

    if num_classes == 2:
        custom_palette = ["#a6cee3", "#1f78b4"]

    elif num_classes == 3:
        custom_palette = ["#1f78b4", "#fb9a99", "#e31a1c"]

    elif num_classes == 4:
        custom_palette = ["#a6cee3", "#1f78b4", "#fb9a99", "#e31a1c"]

    ax = sns.lineplot(data=df_melted, x='epoch', y='pseudo_dice', hue='class', linewidth=3, palette=custom_palette)

    # Get the lines from the plot
    lines = ax.get_lines()

    # Bring the first line to the front by setting a higher zorder
    if num_classes == 3:
        lines[0].set_zorder(5)

    # Left only unique values in contrasts
    contrasts = list(set(contrasts))

    # Make every contrast in upper case:
    contrasts = [contrast.upper() for contrast in contrasts]

    # If 3-cont is in contrasts, rename it to multi-contrast
    if '3CONT' in contrasts:
        contrasts[contrasts.index('3CONT')] = 'Multi-contrast'

    # Generate a title for the plot based on the contrasts
    if len(contrasts) == 1:
        title = f'Validation Pseudo Dice vs. Epoch ({contrasts[0]} model, fold 0, spinal level {spinal_level[0]})'
    elif len(contrasts) > 1:
        # Join contrasts with ' vs ' to create a formatted string
        contrasts_str = ' vs '.join(contrasts)
        title = f'Validation Pseudo Dice vs. Epoch ({contrasts_str} models, fold 0, spinal level {spinal_level[0]})'

    # Fix the y-axis range to be between 0 and 1 and x-axis range to be between 0 and the maximum epoch
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, df_melted['epoch'].max() + 1)

    # Add titles and labels
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Validation Pseudo Dice', fontsize=18)

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('_', ' ').title() for label in labels]  # Modify legend labels
    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=16, title='', frameon=True)

    # Adjust font sizes for all elements
    ax.tick_params(labelsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='18')  # for legend title
    plt.legend(title='', fontsize=11, loc='upper right')

    # Tighten layout
    plt.tight_layout()

    # Save plot to a file
    plt.savefig(fname_fig, dpi=300)
    print(f'Saved plot to {fname_fig}')
    plt.show()


def main():
    """
    Main function to parse the command line arguments and call the plotting function.
    :return:
    """
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    combined_data = []
    log_file_paths = []
    spinal_level = args.spinal_level
    dataset_names = []
    dataset_info = []
    contrasts = []
    columns_with_subjects = []

    # Loop through each log file path
    for log_file_path in args.i:
        log_file_path = os.path.abspath(os.path.expanduser(log_file_path))
        log_file_paths.append(log_file_path)
        data, fold_number = extract_epoch_and_dice(log_file_path)
        df = pd.DataFrame(data)
        df_pseudo_dice = pd.DataFrame(df['pseudo_dice'].to_list()[:-1],
                                      columns=[f'validation_pseudo_dice_class_{i + 1}'
                                               for i in range(len(df['pseudo_dice'].iloc[0]))])
        df = pd.concat([df, df_pseudo_dice], axis=1).drop('pseudo_dice', axis=1)

        # Extract dataset name from the log file path
        dataset_name = log_file_path.split('/')[-4][:10]
        dataset_names.append(dataset_name)

        # Rename df[f'validation_pseudo_dice_class_{spinal_level}'] with adding dataset_name to the column name
        df[f'validation_pseudo_dice_class_{spinal_level[0]}_{dataset_name}'] = df[
            f'validation_pseudo_dice_class_{spinal_level[0]}']

        # Identify columns to drop and drop them
        columns_to_drop = [col for col in df.columns if col.startswith('validation_pseudo_dice_class') and not
        col.endswith(f'{spinal_level[0]}_{dataset_name}')]
        df = df.drop(columns=columns_to_drop)

        # Add columns to combined data
        combined_data.append(df)

    # Combine the data into one df - as columns
    combined_data = pd.concat(combined_data, axis=1)

    # Construct a part of the filename based on all dataset names
    dataset_names_part = '_'.join(dataset_names)

    # Generate the output image filename dynamically
    output_image = f"{args.output_image}/spinal_level_{spinal_level[0]}_validation_pseudo_dice_{dataset_names_part}.png"

    # Remove duplicities (left only the second name of the column)
    combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

    # Extract dataset identifiers, contrast, patch-size info, and number of subjects from log file paths
    for path in log_file_paths:
        dataset_id = path.split('/')[-4].split('_')[0]

        # Extract contrast info:
        contrast = path.split('/')[-4].split('-')[3]
        contrasts.append(contrast)

        # Extract patch size info
        patch_size_label = 'increased patch-size' if 'patchsize' in path else 'default patch-size'

        # Extract number of subjects using regex
        match = re.search(r'(\d+)sub', path)
        if match:
            num_subjects = match.group(1)
        else:
            num_subjects = 'unknown'

        # Create a label with number of subjects and patch size
        dataset_label = f"{num_subjects} training subjects, {patch_size_label}"

        dataset_info.append((dataset_id, dataset_label))

    # Dynamically rename the columns based on the datasets present, subjects, and patch size info
    column_rename_map = {
        f'validation_pseudo_dice_class_{spinal_level[0]}_{dataset_id}': dataset_label
        for dataset_id, dataset_label in dataset_info
    }

    # Rename columns in combined_data
    combined_data = combined_data.rename(columns=column_rename_map)

    # Loop through each column name in the combined_data (excluding 'epoch')
    for column in combined_data.columns:
        if column == 'epoch':
            continue  # Skip the 'epoch' column

        # Extract number of training subjects
        subject_match = re.search(r'(\d+) training subjects', column)

        # Determine patch size (default first, then increased)
        patch_size_priority = 0 if 'default patch-size' in column else 1

        if subject_match:
            num_training_images = int(subject_match.group(1))  # Convert to integer for sorting
            columns_with_subjects.append((num_training_images, patch_size_priority, column))

    # Sort columns first by number of training subjects, then by patch size (default first)
    columns_with_subjects.sort(key=lambda x: (x[0], x[1]))

    # Reconstruct the ordered columns list with 'epoch' first, followed by the sorted columns
    ordered_columns = ['epoch'] + [column for _, _, column in columns_with_subjects]

    # Reorder the combined_data columns
    combined_data = combined_data[ordered_columns]

    # Create the figure
    create_figure(combined_data, output_image, spinal_level, contrasts)


if __name__ == "__main__":
    main()
