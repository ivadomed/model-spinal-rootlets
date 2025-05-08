"""
This script generates a figure showing the distributions of the distances from PMJ for spinal and vertebral levels.
It also computes the mean and standard deviation for each spinal and vertebral level.

Usage: python distributions_and_heights.py -i <input_file> -o <output_folder> -p <participants_file> -normalised <y/n>
"""

import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import os
import argparse


def get_parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate a figures showing statistics distributions of the distances from PMJ for spinal and '
                    'vertebral levels', prog=os.path.basename(__file__).strip('.py'))
    parser.add_argument('-i', required=True, type=str, help='Path to the data_processed folder with CSV '
                        'files with distances from PMJ for spinal and vertebral levels (created by script '
                                                            'generate_figure_rootlets_and_vertebral_spinal_levels.py).')
    parser.add_argument('-o', required=True, type=str, help='Path to the output folder, where figures will '
                                                            'be saved.')
    parser.add_argument('-p', required=True, type=str, help='Path to *.csv file (participants.csv).')
    parser.add_argument('-normalised', required=True, type=str, choices=["y", "n"], help='Choice of '
                        'normalisation by height of subject (yes/no).')

    return parser


def process_data(df, level_type, participants_df, normalised):
    """
    Function to process the data and create a pivot tables for the given level type (rootlets or vertebrae) - for
    analysis of midpoint positions and analysis of levels height.
    :param df: input dataframe with distances from PMJ for spinal and vertebral levels
    :param level_type: type of level to process ('rootlets' or 'vertebrae')
    :return: pivot tables with renamed columns according to anatomical nomenclature
    """
    # merge with participants_df to get height of subjects
    df = pd.merge(df, participants_df, on="subject")
    df_filtered = df[df["spinal_level_type"] == level_type].copy()

    # calculate distance from PMJ to the midpoint of the segment
    if normalised == "y":
        df_filtered["mean_distance_height_pmj"] = (((df_filtered[["distance_from_pmj_start", "distance_from_pmj_end"]].mean(axis=1))/(df_filtered["height_sub"])*df_filtered["height_sub"].median()))
    else:
        df_filtered["mean_distance_height_pmj"] = df_filtered[["distance_from_pmj_start", "distance_from_pmj_end"]].mean(axis=1)

    # make table with positions of midpoint from PMJ for each level
    df_pivot = df_filtered.pivot_table(index="subject", columns="spinal_level", values="mean_distance_height_pmj", aggfunc="mean")
    df_pivot.reset_index(inplace=True)

    # make table with mean and std of height for each segment
    df_mean_std_height = df_filtered.pivot_table(index="subject", columns="spinal_level", values="height",
                                                 aggfunc="mean")
    df_mean_std_height.reset_index(inplace=True)

    # rename columns to be consistent with anatomical nomenclature (spinal levels: C2, C3, C4, C5, C6, C7, C8, T1;
    # vertebral levels: C1, C2, C3, C4, C5, C6, C7, T1)
    if level_type == "vertebrae":
        for i, col in enumerate(df_pivot.columns[1:]):
            if col < 8:
                df_pivot.rename(columns={col: f"Vertebral level C{col}"}, inplace=True)
                df_mean_std_height.rename(columns={col: f"Vertebral level C{col}"}, inplace=True)
            else:
                df_pivot.rename(columns={col: f"Vertebral level T{col - 7}"}, inplace=True)
                df_mean_std_height.rename(columns={col: f"Vertebral level T{col - 7}"}, inplace=True)
    else:
        for i, col in enumerate(df_pivot.columns[1:]):
            if col < 9:
                df_pivot.rename(columns={col: f"Spinal level C{col}"}, inplace=True)
                df_mean_std_height.rename(columns={col: f"Spinal level C{col}"}, inplace=True)
            else:
                df_pivot.rename(columns={col: f"Spinal level T{col - 8}"}, inplace=True)
                df_mean_std_height.rename(columns={col: f"Spinal level T{col - 8}"}, inplace=True)
    return df_pivot, df_mean_std_height


def check_normality(df_pivot):
    """
    Function to check normality of the distributions for each spinal and vertebral level using Shapiro-Wilk test.
    :param df_pivot: dataframe with distances from PMJ for spinal and vertebral levels
    :return: dictionary with results of normality test for each level
    """
    for col in df_pivot.columns[1:]:
        data = df_pivot[col].dropna()
        stat, p_value = shapiro(data)
        if p_value > 0.05:
            print(f"Normal distribution of {col}")
        else:
            print(f"Not normal distribution of {col}")


def compute_mean_std_for_each_level(df_rootlets_pivot, df_vertebrae_pivot, output_path):
    """
    Function to compute the mean and standard deviation for each spinal and vertebral level and save the results to
    CSV files.
    :param df_rootlets_pivot: dataframe with distances from PMJ for spinal levels
    :param df_vertebrae_pivot: dataframe with distances from PMJ for vertebral levels
    :param output_path: path to the output folder where the results will be saved
    :return: mean and standard deviation for each spinal and vertebral level
    """
    # remove first column (subject) from the pivot tables
    df_rootlets_pivot = df_rootlets_pivot.iloc[:, 1:]
    df_vertebrae_pivot = df_vertebrae_pivot.iloc[:, 1:]

    # compute mean and std for each spinal and vertebral level
    mean_rootlets = df_rootlets_pivot.mean(axis=0).round(2)
    std_rootlets = df_rootlets_pivot.std(axis=0).round(2)
    mean_vertebrae = df_vertebrae_pivot.mean(axis=0).round(2)
    std_vertebrae = df_vertebrae_pivot.std(axis=0).round(2)

    # make tables for spinal levels and vertebral levels
    rootlets_table = pd.DataFrame(
        {"Spinal level": mean_rootlets.index, "Mean height": mean_rootlets.values, "Std height": std_rootlets.values})
    vertebrae_table = pd.DataFrame({"Vertebral level": mean_vertebrae.index, "Mean height": mean_vertebrae.values,
                                    "Std height": std_vertebrae.values})

    # save tables to csv
    rootlets_table.to_csv(os.path.join(output_path, "rootlets_size_mean_std.csv"))
    vertebrae_table.to_csv(os.path.join(output_path, "vertebrae_size_mean_std.csv"))

    return mean_rootlets, std_rootlets, mean_vertebrae, std_vertebrae


def plot_distributions(df_rootlets_pivot, df_vertebrae_pivot, output_path, normalised):
    """
    Function to plot the distributions of the distances from PMJ for spinal and vertebral levels
    (approximated by PDF functions).
    :param df_rootlets_pivot: dataframe with distances of midpoints from PMJ for spinal levels
    :param df_vertebrae_pivot: dataframe with distances of midpoints from PMJ for vertebral levels
    :param output_path: path to the output folder where the figure will be saved
    :param normalised: normalisation by height of subject (yes/no)
    :return:
    """
    # define parameters for the plot
    x_limit = 200
    y_limit = 0.25
    step = 0.05
    x = np.linspace(0, x_limit, 2000)

    # define the custom color palette
    custom_palette = sns.color_palette("tab10", 10)
    custom_palette[6] = "#f14cc1"
    custom_palette[7] = "black"
    sns.set_palette("Set2")

    # combine the two dataframes for plotting with parameters of line style
    dataframes = zip([df_rootlets_pivot, df_vertebrae_pivot], ["-", "--"])

    # create a figure
    plt.figure(figsize=(10, 6))

    # plot the distributions for each dataframe
    for df, style in dataframes:
        for i, col in enumerate(df.columns[1:]):
            data = df[col].dropna()
            mu, sigma = norm.fit(data)
            y = norm.pdf(x, mu, sigma)
            plt.plot(x, y, label=col, linestyle=style, color=custom_palette[i % len(custom_palette)])
            plt.xlim(0, x_limit)
            plt.ylim(0, y_limit)
            plt.yticks(np.arange(0, y_limit + 0.01, step), fontsize=18)

    # set the title and labels according to the normalisation
    if normalised == "y":
        plt.xlabel(
            "Distribution of the midpoint from the PMJ distances [mm] \n (normalised by subject height and scaled by median height)",
            fontsize=18)
        normalised_label = "normalised"
    else:
        plt.xlabel(
            "Distribution of the midpoint from the PMJ distances [mm] \n (not normalised by subject height)",
            fontsize=18)
        normalised_label = "not-normalised"

    # set the legend and grid
    plt.legend(ncol=2, fontsize=15)
    plt.ylabel("Probability Density Function [-]", fontsize=20)
    plt.xticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, f"distributions_120_sub_{normalised_label}_height.svg"),
                    dpi=300)
    plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get data from the command line arguments
    file_path = os.path.abspath(args.i)
    output_path = os.path.abspath(args.o)
    participants_path = os.path.abspath(args.p)
    normalised = args.normalised

    # Load the data
    df = pd.read_csv(file_path)
    participants_df = pd.read_csv(participants_path, sep="\t")

    # Process the data and create pivot tables
    df_rootlets_pivot, df_mean_std_height_rootlets = process_data(df, "rootlets", participants_df, normalised)
    df_vertebrae_pivot, df_mean_std_height_vertebrae = process_data(df, "vertebrae", participants_df, normalised)

    # Check normality of the distributions
    check_normality(df_rootlets_pivot)
    check_normality(df_vertebrae_pivot)

    # Plot the distributions of the distances from PMJ for spinal and vertebral levels
    plot_distributions(df_rootlets_pivot, df_vertebrae_pivot, output_path, normalised)

    # Compute mean and standard deviation for each spinal and vertebral level
    compute_mean_std_for_each_level(df_mean_std_height_rootlets, df_mean_std_height_vertebrae, output_path)


if __name__ == "__main__":
    main()