"""
The script reads the CSV files with Dice coefficient (and other segmentation metrics) computed using
[MetricReloaded](https://github.com/ivadomed/MetricsReloaded/blob/main/compute_metrics_reloaded.py) from the BIDS
structured data folder, creates a combined dataframe, and then creates a boxplot/violinplot from the data.
The segmentation metrics can be computed by the script get_statistics_hc-leipzig-7t-mp2rage.py.

Usage: python dice-score-graph.py -i /path/to/BIDS_structured_data -dataset-folds Dataset001_fold0 Dataset002_fold0 
-contrast UNIT1 -output-combined-csv /path/to/output_folder -plot-type boxplot

"""

import os
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
import seaborn as sns
import matplotlib.pyplot as plt


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script makes boxplots/violinplots from data structured in BIDS.',
        formatter_class=RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-i',
        required=True,
        help='Path to the BIDS structured data folder.'
    )

    parser.add_argument(
        '-dataset-folds',
        required=True,
        nargs='+',
        help='Dataset and fold names - e.g. Dataset001_fold0, Dataset002_fold0.'
    )

    parser.add_argument(
        '-contrast',
        required=False,
        help='Name of contrast, that you want to visualize.',
        choices=['UNIT1', 'UNIT1_neg', 'inv-1_part-mag_MP2RAGE', 'inv-1_part-mag_MP2RAGE_neg',
                 'inv-2_part-mag_MP2RAGE', 'inv-2_part-mag_MP2RAGE_neg']
    )

    parser.add_argument(
        '-output-combined-csv',
        required=False,
        help='Path to the output folder, where you want to save the combined CSV file'

    )

    parser.add_argument(
        '-plot-type',
        required=False,
        choices=['boxplot', 'violinplot'],
        help='Type of plot you want to create. Default is boxplot.',
        default='boxplot'

    )
    return parser


def create_dataframe(directory, dataset_folds, analysed_contrast, output_combined_csv, contrast):

    """
    This function reads the CSV files with statistical data from the BIDS structured data folder, creates a combined
    dataframe and returns it.
    :param directory: Path to the BIDS structured data folder.
    :param dataset_folds: Dataset and fold names - e.g. Dataset001_fold0, Dataset002_fold0.
    :param analysed_contrast: Name of contrast, that you want to visualize.
    :param output_combined_csv: Path to the output folder, where you want to save the combined CSV file.
    :param contrast: Name of contrast, that you want to visualize.
    :return: Combined dataframe.
    """

    dataframes = []
    contrast_value = []

    for file in os.listdir(directory):
        subject_dir = os.path.join(directory, file, 'anat')
        if not os.path.exists(subject_dir):
            continue

        # Iterate over each file in the directory
        for filename in os.listdir(subject_dir):
            for dataset_fold in dataset_folds:
                # Note: the CSV files were generated using the compute_metrics_reloaded.py script:
                #  https://github.com/ivadomed/MetricsReloaded/blob/main/compute_metrics_reloaded.py
                if filename.endswith(f'{dataset_fold}_metrics.csv') and contrast in filename:
                    # Extract subject, dataset, and fold from the filename
                    parts = filename.split('_')
                    subject = parts[1]  # Assuming subject ID is the second part
                    dataset = parts[-3]  # Assuming dataset is the second to last part (e.g., Dataset011)
                    fold = parts[-2]  # Assuming fold is the last part (e.g., fold0)
                    contrast_value.append(contrast)

                    # Read the CSV file
                    df = pd.read_csv(os.path.join(subject_dir, filename))

                    # Add columns for subject and dataset
                    df['subject'] = subject
                    df['dataset'] = dataset+'_'+fold
                    df['contrast'] = contrast_value[0]
                    contrast_value = []

                    # Append dataframe to list
                    dataframes.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Add 0.005 to the DiceSimilarityCoefficient column to avoid zero values (to be able to plot the data)
    combined_df['DiceSimilarityCoefficient'] = combined_df['DiceSimilarityCoefficient'].apply(lambda x: x + 0.005 if x == 0 else x)

    # Make column label as only integer value
    combined_df['label'] = combined_df['label'].astype(int)

    # Extract unique dataset and fold combinations from the 'dataset' column
    unique_datasets = sorted(combined_df['dataset'].unique())

    # Convert the 'dataset' column to a categorical type with the dynamically generated order
    combined_df['dataset'] = pd.Categorical(combined_df['dataset'], categories=unique_datasets, ordered=True)

    # Sort the DataFrame by the 'dataset' column
    combined_df = combined_df.sort_values('dataset')

    # Remove columns EmptyPred and EmptyRef
    combined_df = combined_df.drop(columns=['EmptyPred', 'EmptyRef'])

    # Left in the table only rows with wanted contrast
    combined_df = combined_df[combined_df['contrast'] == analysed_contrast]

    # Save the combined dataframe to a new CSV file if the output_combined_csv argument is provided
    if output_combined_csv:
        combined_df.to_csv(f"{output_combined_csv}/combined-dataframe.csv", index=False)
        print(f'Saved the combined dataframe to {output_combined_csv}')

    return combined_df


def create_boxplot_or_violinplot(combined_df, plot_type, contrast):

    """
    This function creates a boxplot or violinplot from the combined dataframe.
    :param combined_df: Combined dataframe.
    :param plot_type: Type of plot you want to create.
    :param contrast: Name of contrast, that you want to visualize.
    """

    plt.figure(figsize=(12, 6))
    sns.set(style="dark")

    if plot_type == 'boxplot':
        sns.boxplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df, hue='dataset')

    elif plot_type == 'violinplot':
        sns.violinplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df, hue='dataset')

    plt.title(f'Dice score between different datasets and folds (contrast: {contrast})', fontweight='bold')
    plt.xlabel('Spinal level')
    plt.ylabel('Dice Similarity Coefficient')
    plt.legend(title='')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    directory = args.i
    dataset_folds = args.dataset_folds
    analysed_contrast = args.contrast
    output_combined_csv = args.output_combined_csv
    plot_type = args.plot_type

    # Create the combined dataframe
    combined_dataframe = create_dataframe(directory, dataset_folds, analysed_contrast, output_combined_csv,
                                          analysed_contrast)

    # Create the boxplot/violinplot
    create_boxplot_or_violinplot(combined_dataframe, plot_type, analysed_contrast)


if __name__ == "__main__":
    main()


