import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from scipy import stats
from matplotlib import gridspec

CUSTOM_PALETTE = sns.color_palette("tab10", 7)
CUSTOM_PALETTE[6] = "#f14cc1"


def get_parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate a Bland-Altman and correlation plot in one figure.')
    parser.add_argument('-i', required=True, type=str, help='Path to the CSV file with distances from PMJ.')
    parser.add_argument('-o', required=True, type=str, help='Path to the output folder.')
    parser.add_argument('-p', required=True, type=str, help='Path to participants.csv file.')
    return parser


def process_data(df, level_type, df_participants):
    """
    Function to process the data for spinal and vertebral levels.
    :param df: Input data frame with distances from PMJ
    :param level_type: level type (either "rootlets" or "vertebrae")
    :param df_participants: dataframe of participants with their heights
    :return: prepared dataframe for analysis
    """

    # Merge data with participant's data (the height will be used for normalization)
    df = pd.merge(df, df_participants, on="subject")

    # Filter data for the specified level type and calculate the position of each midpoint
    df_filtered = df[df["spinal_level_type"] == level_type].copy()
    df_filtered["mean_distance"] = df_filtered[["distance_from_pmj_start", "distance_from_pmj_end"]].mean(axis=1)

    # Normalise the distance of midpoint to the PMJ with the height of the subject and scale it to the median height
    # (to preserve units)
    df_filtered["mean_distance_height"] = (df_filtered["mean_distance"] / (df_filtered["height_sub"])) * (
        df_filtered["height_sub"].median())

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

    return df_pivot


def check_normality(bland_altman_dataframe, global_local):
    """
    Function to check the normality of the data using the Shapiro-Wilk test.
    :param bland_altman_dataframe: dataframe with data for Bland-Altman analysis
    :param global_local: variant to check normality for all data or by level (local or global)
    :return: None
    """
    # Check normality of the differences for each pair of spinal and vertebral levels separately
    if global_local == "local":
        print("\nShapiro-Wilk Test for Normality by Level:")
        for level, data in bland_altman_dataframe.groupby("Level")["Difference"]:
            stat, p = shapiro(data)
            print(f"Level {level}: Statistics={stat:.3f}, p={p:.3f}")
            print(f"Level {level} is {'NOT ' if p < 0.05 else ''}normally distributed")

    # Check normality of the differences for all pairs of spinal and vertebral levels together
    else:
        print("\nShapiro-Wilk Test for Normality:")
        data = bland_altman_dataframe["Difference"]
        stat, p = shapiro(data)
        print(f"Statistics={stat:.3f}, p={p:.3f}")
        print(f"The data is {'NOT ' if p < 0.05 else ''}normally distributed")


def create_data_regression_bland_altman(df_spinal, df_vertebral, levels_for_comparison, spinal_shift):
    """
    Function to create dataframes for regression and Bland-Altman analysis.
    :param df_spinal: dataframe with spinal levels
    :param df_vertebral: dataframe with vertebral levels
    :param levels_for_comparison: list of levels for comparison
    :param spinal_shift: shift for spinal levels (0 or 1)
    :return: dataframes for regression and Bland-Altman analysis
    """
    data_regression = []
    data_bland_altman = []

    # Create dataframes for regression and Bland-Altman analysis
    for i in range(1, len(df_vertebral.columns)):
        spinal_vals = df_spinal.iloc[:, i + spinal_shift].to_numpy()
        vertebral_vals = df_vertebral.iloc[:, i].to_numpy()
        level_name = levels_for_comparison[i - 1]
        mean_values = (spinal_vals + vertebral_vals) / 2
        differences = vertebral_vals - spinal_vals

        # Create data for regression analysis
        for sp, vert in zip(spinal_vals, vertebral_vals):
            data_regression.append({
                "Spinal": sp,
                "Vertebral": vert,
                "Level": level_name
            })
        # Create data for Bland-Altman analysis
        for mean, diff, sp, vert in zip(mean_values, differences, spinal_vals, vertebral_vals):
            data_bland_altman.append({
                "Mean": mean, "Difference": diff, "Level": level_name,
                "Spinal": sp, "Vertebral": vert
            })

    regression_dataframe = pd.DataFrame(data_regression)
    bland_altman_dataframe = pd.DataFrame(data_bland_altman)
    return regression_dataframe, bland_altman_dataframe


def regression_and_bland_altman_plot(df_spinal, df_vertebral, output_path, shifted, local_global="global"):
    """
    Function to compute regression analysis and create a Bland-Altman plot.
    :param df_spinal: dataframe with spinal levels
    :param df_vertebral: dataframe with vertebral levels
    :param output_path: path where the output figure will be saved
    :param shifted: shifted (True) or not shifted (False) variant
    :return: None
    """
    position_x = 197

    # Check if the analysis should be done for 'shifted' variant (i.e., VLC2-SLC3, VLC3-SLC4, etc.) or
    # 'not-shifted' variant (i.e., VLC2-SLC2, VLC3-SLC3, etc.) and set the parameters accordingly
    if shifted:
        shifted_name = "shifted"
        spinal_shift = 1
        text_shift_lower_level = 3.6
        levels_for_comparison = ["VLC2-SLC3", "VLC3-SLC4", "VLC4-SLC5", "VLC5-SLC6", "VLC6-SLC7", "VLC7-SLC8",
                                 "VLT1-SLT1"]
        y_limit_start = -25
        y_limit_end = 25
        first_percentile = 13.2
        second_percentile = -6.5
        median_value = 2.7
    else:
        shifted_name = "not_shifted"
        spinal_shift = 0
        text_shift_lower_level = 2.9
        levels_for_comparison = ["VLC2-SLC2", "VLC3-SLC3", "VLC4-SLC4", "VLC5-SLC5", "VLC6-SLC6", "VLC7-SLC7",
                                 "VLT1-SLC8"]
        y_limit_start = -10
        y_limit_end = 40
        first_percentile = 27.5
        second_percentile = 8.7
        median_value = 17.7

    # Create dataframes for regression and Bland-Altman analysis
    regression_dataframe, bland_altman_dataframe = create_data_regression_bland_altman(df_spinal, df_vertebral,
                                                                                    levels_for_comparison, spinal_shift)
    if local_global == "global":
        # Make linear regression and correlation
        corr, p = stats.spearmanr(regression_dataframe["Spinal"], regression_dataframe["Vertebral"])
        print(f"Spearman correlation: {corr:.3f}, p-value: {p:.3f}")
        slope, intercept, r_value, p_value, std_err = stats.linregress(regression_dataframe["Spinal"],
                                                                       regression_dataframe["Vertebral"])
        print(f"Linear regression: y = {slope:.3f}x + {intercept:.3f}, R2 = {r_value ** 2:.3f}")

        # Check normality of the differences (Shapiro-Wilk test)
        check_normality(bland_altman_dataframe, "global")

        # Calculate the median difference and the 2.5th and 97.5th percentiles
        mean_diff = bland_altman_dataframe["Difference"].median()
        lower_limit = np.percentile(bland_altman_dataframe["Difference"], 2.5)
        upper_limit = np.percentile(bland_altman_dataframe["Difference"], 97.5)

        # Set the colour palette for the plot
        palette_dict = dict(zip(levels_for_comparison, CUSTOM_PALETTE))

        # Set parameters for the global plot
        fig = plt.figure(figsize=(13, 5.4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2.8, 3])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Create the scatterplot and regression line for spinal and vertebral levels
        sns.scatterplot(x="Spinal", y="Vertebral", hue="Level", data=regression_dataframe, ax=ax1, alpha=0.5,
                        palette=CUSTOM_PALETTE, legend=False, edgecolor=None)
        sns.regplot(x="Spinal", y="Vertebral", data=regression_dataframe, ax=ax1, scatter=False,
                    line_kws={"color": "black", "linewidth": 2}, ci=95)

        # Reference line (y=x)
        ax1.plot([0, 200], [0, 200], color="gray", linestyle="--", alpha=0.8)

        # Add text with the regression equation and R-squared value
        if intercept < 0:
            ax1.text(65, 37, f"y = {slope:.2f}x - {abs(intercept):.2f}\n$R^2$ = {r_value ** 2:.2f}", fontsize=16)
        else:
            ax1.text(56, 35, f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value ** 2:.2f}", fontsize=16)

        # Define parameters for the subplot
        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='both', labelsize=14)
        ax1.set_xlim(0, 200)
        ax1.set_ylim(0, 200)
        ax1.set_yticks(np.arange(0, 201, 50))
        ax1.set_xlabel("Normalised spinal level distance [mm]", fontsize=16)
        ax1.set_ylabel("Normalised vertebral level distance [mm]", fontsize=16)
        ax1.set_title("Regression analysis", fontsize=18)
        ax1.grid(True, alpha=0.5)

        # Bland-Altman plot (scatterplot with mean and difference and median and percentiles lines)
        sns.scatterplot(data=bland_altman_dataframe, x="Mean", y="Difference", hue="Level",
                        palette=palette_dict, alpha=0.5, ax=ax2, edgecolor=None)
        ax2.axhline(mean_diff, color='black', linestyle="dashed", linewidth=2)
        ax2.axhline(upper_limit, color='black', linestyle="--", linewidth=1)
        ax2.axhline(lower_limit, color='black', linestyle="--", linewidth=1)
        ax2.text(position_x, median_value, 'Median', ha="right", va="bottom", fontsize=15, color="black")
        ax2.text(position_x, median_value - 1, f'{mean_diff:.2f}', ha="right", va="top", fontsize=15, color="black")
        ax2.text(position_x, first_percentile, '97.5 percentile', ha="right", va="bottom", fontsize=15)
        ax2.text(position_x, first_percentile - 1, f'{upper_limit:.2f}', ha="right", va="top", fontsize=15)
        ax2.text(position_x, second_percentile, '2.5 percentile', ha="right", va="bottom", fontsize=15)
        ax2.text(position_x, second_percentile - 1, f'{lower_limit:.2f}', ha="right", va="top", fontsize=15)

        # Define parameters for the subplot
        ax2.set_xlabel("Mean distance from PMJ [mm]", fontsize=16)
        ax2.set_ylabel("Difference [mm]", fontsize=16)
        ax2.set_title("Bland-Altman plot", fontsize=18)
        ax2.set_xlim(0, 200)
        ax2.set_ylim(y_limit_start, y_limit_end)
        ax2.set_xticks(np.arange(0, 201, 25))
        ax2.set_yticks(np.arange(y_limit_start, y_limit_end + 1, 5))
        ax2.legend(loc="lower left", bbox_to_anchor=(0.98, 0.37), fontsize=17, frameon=False, handletextpad=0.3)
        ax2.yaxis.grid(True, alpha=0.5)
        ax2.xaxis.grid(False)

        # Make space for the legend
        plt.subplots_adjust(right=0.8)
        plt.tight_layout()

        # Save the figure as svg
        plt.savefig(os.path.join(output_path, f"global_bland_altman_{shifted_name}.svg"),
                    bbox_inches='tight')
        plt.show()

    else:
        # Check normality of the differences (Shapiro-Wilk test) locally for each pair of spinal and vertebral levels
        check_normality(bland_altman_dataframe, "local")

        # Create a figure with subplots for each spinal and vertebral level
        fig = plt.figure(figsize=(30, 28))
        gs = gridspec.GridSpec(4, 5, width_ratios=[2, 2, 0.1, 2, 2])

        # Define the parameters for each level for the subplots
        level_params = [
            {"pos": 0, "xlim": 10, "ylim": 90, "text_x": 14, "text_y": 65},
            {"pos": 5, "xlim": 30, "ylim": 110, "text_x": 34, "text_y": 80},
            {"pos": 10, "xlim": 50, "ylim": 130, "text_x": 54, "text_y": 100},
            {"pos": 15, "xlim": 60, "ylim": 140, "text_x": 64, "text_y": 115},
            {"pos": 3, "xlim": 70, "ylim": 150, "text_x": 74, "text_y": 130},
            {"pos": 8, "xlim": 80, "ylim": 160, "text_x": 84, "text_y": 140},
            {"pos": 13, "xlim": 100, "ylim": 180, "text_x": 104, "text_y": 165},
        ]

        # For each subplot define the parameters
        for i, (level, params) in enumerate(zip(levels_for_comparison, level_params)):
            colour = CUSTOM_PALETTE[i]
            pos = params["pos"]
            x_limit_reg = params["xlim"]
            y_limit_reg = params["ylim"]
            text_x = params["text_x"]
            text_y = params["text_y"]

            ax1 = fig.add_subplot(gs[pos])
            ax2 = fig.add_subplot(gs[pos + 1])
            data = bland_altman_dataframe[bland_altman_dataframe["Level"] == level]
            spinal_vals = data["Spinal"]
            vertebral_vals = data["Vertebral"]
            mean_values = data["Mean"]
            differences = data["Difference"]

            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(spinal_vals, vertebral_vals)

            # Calculate the mean difference and the 2.5th and 97.5th percentiles for the Bland-Altman plot
            median_diff = differences.median()
            lower_limit = np.percentile(differences, 2.5)
            upper_limit = np.percentile(differences, 97.5)

            x_center = mean_values.mean()
            x_min = round(x_center - 35, -1)
            x_max = round(x_center + 45, -1)

            # Create the regression plot
            sns.regplot(x=spinal_vals, y=vertebral_vals, ax=ax1, color=colour, line_kws={"color": "black"},
                        scatter_kws={'alpha': 0.6, 's': 100})
            ax1.plot([x_limit_reg, y_limit_reg], [x_limit_reg, y_limit_reg], color="black", linestyle="--")
            ax1.set_xlim(x_limit_reg, y_limit_reg)
            ax1.set_ylim(x_limit_reg, y_limit_reg)
            ax1.set_xticks(np.arange(x_limit_reg, y_limit_reg + 1, 20))
            ax1.set_yticks(np.arange(x_limit_reg, y_limit_reg + 1, 20))
            ax1.set_xlabel("Normalised SL distances [mm]", fontsize=27)
            ax1.set_ylabel("Normalised VL distances [mm]", fontsize=27)
            ax1.set_title(f"Regression: {level}", fontsize=27)
            ax1.tick_params(axis='both', labelsize=25)
            ax1.text(text_x, text_y, f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value ** 2:.2f}", fontsize=25)
            ax1.grid(True, alpha=0.5)

            # Create the Bland-Altman plot
            sns.scatterplot(x=mean_values, y=differences, color=CUSTOM_PALETTE[i], edgecolor=None, alpha=0.7, ax=ax2,
                            s=110)
            ax2.axhline(median_diff, color='black', linestyle="dashed", linewidth=2)
            ax2.axhline(upper_limit, color='black', linestyle="dashed", linewidth=1)
            ax2.axhline(lower_limit, color='black', linestyle="dashed")
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_limit_start, y_limit_end)
            ax2.set_xticks(np.arange(x_min, x_max + 1, 20))
            ax2.set_yticks(np.arange(y_limit_start, y_limit_end + 1, 10))

            ax2.text(x_max - 2, median_diff + 0.3, f"M: {median_diff:.2f}", ha="right", va="bottom", fontsize=25)
            ax2.text(x_max - 2, upper_limit + 0.35, f"97.5 P: {upper_limit:.2f}", ha="right", va="bottom", fontsize=25)
            ax2.text(x_max - 2, lower_limit + text_shift_lower_level, f"2.5 P: {lower_limit:.2f}", ha="right", va="top",
                     fontsize=25)

            ax2.set_title(f"Bland-Altman: {level}", fontsize=27)
            ax2.set_xlabel("Mean distance from PMJ [mm]", fontsize=27)
            ax2.set_ylabel("Difference VL - SL [mm]", fontsize=27)

            ax2.yaxis.grid(True, alpha=0.5)
            ax2.xaxis.grid(False)
            ax2.tick_params(axis='both', labelsize=25)

        plt.tight_layout(h_pad=3.0, w_pad=2.5)
        plt.savefig(os.path.join(output_path, f"local_regression_and_bland_altman_{shifted_name}.svg"))
        plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get data from the command line arguments
    df = pd.read_csv(os.path.abspath(args.i))
    output_path = os.path.abspath(args.o)
    participants_path = os.path.abspath(args.p)

    df_participants = pd.read_csv(participants_path, sep="\t")

    # Process data for spinal and vertebral levels
    df_spinal = process_data(df, "rootlets", df_participants)
    df_vertebral = process_data(df, "vertebrae", df_participants)

    # Perform regression analysis and create "global" Bland-Altman plot for both shifted and not shifted variants
    regression_and_bland_altman_plot(df_spinal, df_vertebral, output_path, shifted=False, local_global="global")
    regression_and_bland_altman_plot(df_spinal, df_vertebral, output_path, shifted=True, local_global="global")

    # Perform regression analysis and create "local" Bland-Altman plot for both shifted and not shifted variants
    # (for each pair of spinal and vertebral levels)
    regression_and_bland_altman_plot(df_spinal, df_vertebral, output_path, shifted=False, local_global="local")
    regression_and_bland_altman_plot(df_spinal, df_vertebral, output_path, shifted=True, local_global="local")


if __name__ == "__main__":
    main()
