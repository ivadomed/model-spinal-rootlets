import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse


# This script generates one Bland-Altman plot for all pairs of spinal and vertebral levels.
# NOTE: This script will be used for presentation to generate a figure instead of 'bland-altman-separately.py' script,
# which generates subplots for each pair for analysis.


def get_parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate a Bland-Altman and correlation plot in one figure.')
    parser.add_argument('-i', required=True, type=str, help='Path to the CSV file with distances from PMJ.')
    parser.add_argument('-o', required=True, type=str, help='Path to the output folder.')
    parser.add_argument('-p', required=True, type=str, help='Path to participants.csv file.')
    parser.add_argument('-normalise', required=True, type=str, help='Normalise distances to the height of the subject (y/n).')
    return parser


def process_data(df, level_type, participants_path, normalise):
    """
    Function to process the data for spinal and vertebral levels.
    :param df: Input data frame with distances from PMJ
    :param level_type: level type (either "rootlets" or "vertebrae")
    :param participants_path: path to the participants.csv file
    :return: prepared dataframe for Bland-Altman analysis
    """

    # Merge data with participant's data (the height will be used for normalization)
    df_participants = pd.read_csv(participants_path, sep="\t")
    df = pd.merge(df, df_participants, on="subject")

    # Filter data for the specified level type and calculate the position of each midpoint
    df_filtered = df[df["spinal_level_type"] == level_type].copy()
    df_filtered["mean_distance"] = df_filtered[["distance_from_pmj_start", "distance_from_pmj_end"]].mean(axis=1)

    if normalise.lower() == 'y':
        # Normalise the distance of midpoint to the PMJ with the height of the subject and scale it to the median height
        # (to preserve units)
        df_filtered["mean_distance_height"] = (df_filtered["mean_distance"] / (df_filtered["height_sub"])) * (
        df_filtered["height_sub"].median())
        LABEL = "normalised"
    else:
        # Use the mean distance without normalization
        df_filtered["mean_distance_height"] = df_filtered["mean_distance"]
        LABEL = "not_normalised"

    # Make a table, where each row is a subject and each column is a spinal level
    df_pivot = df_filtered.pivot_table(index="subject", columns="spinal_level", values="mean_distance_height",
                                       aggfunc="mean")
    df_pivot.reset_index(inplace=True)

    # Rename the columns (spinal and vertebral levels)
    if level_type == "vertebrae":
        level_prefix = "Vertebral level"
    else:
        level_prefix = "Spinal level"

    # Rename the columns to be consistent with anatomical nomenclature (spinal levels: C2, C3, C4, C5, C6, C7, C8, T1;
    # vertebral levels: C2, C3, C4, C5, C6, C7, T1)
    for col in df_pivot.columns[1:]:
        if level_prefix == "Vertebral level":
            new_name = f"{level_prefix} C{col}" if col < 8 else f"{level_prefix} T{col - 7}"
        else:
            new_name = f"{level_prefix} C{col}" if col < 9 else f"{level_prefix} T{col - 8}"
        df_pivot.rename(columns={col: new_name}, inplace=True)

    # Export the table as a CSV file
    output_csv_path = f"{level_type}_distances_{LABEL}.csv"
    df_pivot.to_csv(output_csv_path, index=False)

    return df_pivot, LABEL


def create_bland_altman(df_spinal, df_vertebral, output_path, shifted, label):
    """
    Function to create a Bland-Altman plot for spinal and vertebral level correspondence.
    :param df_spinal: dataframe with spinal levels for a comparison
    :param df_vertebral: dataframe with vertebral levels for a comparison
    :param output_path: output path for saving the plot
    :param shifted: shifted/not-shifted variant (True/False)
    :return:
    """

    # Check if the analysis should be done for 'shifted' variant (i.e., VLC2-SLC3, VLC3-SLC4, etc.) or
    # 'not-shifted' variant (i.e., VLC2-SLC2, VLC3-SLC3, etc.) and set the parameters accordingly
    if shifted:
        spinal_shift = 1
        levels_for_comparison = {"C2": "VLC2-SLC3", "C3": "VLC3-SLC4", "C4": "VLC4-SLC5", "C5": "VLC5-SLC6", "C6": "VLC6-SLC7",
                         "C7": "VLC7-SLC8", "C8": "VLT1-SLT1"}
        y_limit_start = -25
        y_limit_end = 25
        shifted_name = "shifted"

        text_positions_x = [47.5, 63.7, 79.9, 96.1, 112.3, 128.5, 144.7]
        text_position_y = -14.2

    else:
        spinal_shift = 0
        levels_for_comparison = {"C2": "VLC2-SLC2", "C3": "VLC3-SLC3", "C4": "VLC4-SLC4", "C5": "VLC5-SLC5", "C6": "VLC6-SLC6",
                         "C7": "VLC7-SLC7", "C8": "VLT1-SLC8"}
        y_limit_start = -10
        y_limit_end = 40
        text_positions_x = [38.5, 54.7, 70.9, 87.1, 103.3, 119.5, 135.7]
        text_position_y = 0.8
        shifted_name = "not_shifted"

    # Define figure size
    fig, ax = plt.subplots(figsize=(11, 6))

    # Prepare the data for plotting (mean and difference between spinal and vertebral level for each subject)
    for i in range(1, len(df_vertebral.columns)):
        level_name = df_spinal.columns[i].replace("Spinal level", "").strip()
        spinal_vals = df_spinal.iloc[:, i + spinal_shift].to_numpy()
        vertebral_vals = df_vertebral.iloc[:, i].to_numpy()
        mean_values = (spinal_vals + vertebral_vals) / 2
        differences = vertebral_vals - spinal_vals

        df_plot = pd.DataFrame({
            "Mean": mean_values,
            "Difference": differences,
            "Level": level_name,
            "Spinal": spinal_vals,
            "Vertebral": vertebral_vals
        })

        df_plot["Level"] = df_plot["Level"].replace(levels_for_comparison)

        current_level = df_plot["Level"].unique()[0]

        # Check normality of the differences (Shapiro-Wilk test)
        stat, p = shapiro(df_plot["Difference"])
        print(f"Shapiro-Wilk test for pair {current_level}: p-value={p:.5f}")
        if p < 0.05:
            print("The data is not normally distributed.")
        else:
            print("The data is normally distributed.")

        # calculate the median difference and the 2.5th and 97.5th percentiles
        median_diff = df_plot["Difference"].median()
        median_diff = round(median_diff, 2)
        if median_diff == -0.00:
            median_diff = 0.0
        lower_limit = np.percentile(df_plot["Difference"], 2.5)
        upper_limit = np.percentile(df_plot["Difference"], 97.5)

        # define the color for the current level pair
        custom_palette = [sns.color_palette("tab10")[i - 1]]
        color = custom_palette[0]
        if i == 7:
            custom_palette = ["#f14cc1"]

        # Plot the scatterplot (each dot represents a subject)
        sns.scatterplot(data=df_plot, x="Mean", y="Difference", hue="Level",
                        palette=custom_palette, alpha=0.5, ax=ax)
        # Plot the lines for the mean difference and the percentiles (for each level pair)
        range_x = np.linspace(df_plot["Mean"].median() - 10, df_plot["Mean"].median() + 10, 120)
        sns.lineplot(x=range_x, y=median_diff * np.ones_like(range_x), color="black", linestyle="--", ax=ax, linewidth=2,
                     alpha=0.8)
        sns.lineplot(x=range_x, y=upper_limit * np.ones_like(range_x), color=color, linestyle="--", ax=ax, linewidth=2)
        sns.lineplot(x=range_x, y=lower_limit * np.ones_like(range_x), color=color, linestyle="--", ax=ax, linewidth=2)

        # Add text labels for the mean difference and the percentiles
        ax.text(df_plot["Mean"].median() - 4, median_diff + 1, f"{median_diff:.2f}", fontsize=14, color="black", alpha=0.9,
                fontweight="bold")
        ax.text(df_plot["Mean"].median() - 4, upper_limit + 1, f"{upper_limit:.2f}", fontsize=14, color=color)
        ax.text(df_plot["Mean"].median() - 4, lower_limit - 2, f"{lower_limit:.2f}", fontsize=14, color=color)

        # Add text labels for compared levels with rectangles around them
        ax.text(text_positions_x[i - 1], text_position_y, df_plot["Level"][i], fontsize=11, color=color, alpha=1,
                fontweight="bold")
        rect = plt.Rectangle((text_positions_x[i - 1] - 0.8, text_position_y - 0.3), 15.2, 2, linewidth=1,
                             edgecolor=color, facecolor=color, alpha=0.1)
        ax.add_patch(rect)

    # Set the title, labels, and legend parameters
    ax.set_xlabel("Mean distance from PMJ [mm]", fontsize=18)
    ax.set_ylabel("Difference (vertebral - spinal level) [mm]", fontsize=18)
    ax.set_title("Bland-Altman plot", fontsize=20)
    ax.set_ylim(y_limit_start, y_limit_end)
    ax.set_xlim(25, 175)
    ax.set_xticks(np.arange(25, 176, 25))
    ax.set_yticks(np.arange(y_limit_start, y_limit_end + 1, 5))
    ax.tick_params(axis='both', labelsize=16)
    ax.yaxis.grid(True, alpha=0.5)
    ax.xaxis.grid(False)
    plt.tight_layout()


    # delete legend
    ax.legend_.remove()

    # Save the plot as a PNG file
    plt.savefig(os.path.join(output_path, f"zkcorrelation_and_bland_altman_combined_{shifted_name}_{label}.png"),
                bbox_inches='tight')
    plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # get the variables from the command line
    df = pd.read_csv(os.path.abspath(args.i))
    output_path = os.path.abspath(args.o)
    participants_path = os.path.abspath(args.p)
    normalise = args.normalise

    # Process the data for spinal and vertebral levels
    df_spinal, label = process_data(df, "rootlets", participants_path, normalise)
    df_vertebral, label = process_data(df, "vertebrae", participants_path, normalise)

    # Create Bland-Altman plots for both shifted and not-shifted variants
    create_bland_altman(df_spinal, df_vertebral, output_path, shifted=False, label=label)
    create_bland_altman(df_spinal, df_vertebral, output_path, shifted=True, label=label)


if __name__ == "__main__":
    main()
