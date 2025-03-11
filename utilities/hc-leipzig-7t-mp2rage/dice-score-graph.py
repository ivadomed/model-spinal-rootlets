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
from matplotlib.ticker import MultipleLocator


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
                 'inv-2_part-mag_MP2RAGE', 'inv-2_part-mag_MP2RAGE_neg'],
        nargs='+'
    )

    parser.add_argument(
        '-output',
        required=False,
        help='Path to the output folder, where you want to save the combined CSV file and images'

    )

    parser.add_argument(
        '-plot-type',
        required=False,
        choices=['boxplot', 'violinplot'],
        help='Type of plot you want to create. Default is boxplot.',
        default='boxplot'

    )
    return parser


def create_dataframe(directory, dataset_folds, output, contrast):

    """
    This function reads the CSV files with statistical data from the BIDS structured data folder, creates a combined
    dataframe and returns it.
    :param directory: Path to the BIDS structured data folder.
    :param dataset_folds: Dataset and fold names - e.g. Dataset001_fold0, Dataset002_fold0.
    :param output: Path to the output folder, where you want to save the combined CSV file.
    :param contrast: Name of contrast, that you want to visualize.
    :return: Combined dataframe.
    """

    dataframes = []
    contrast_value = []
    file_names = []

    for file in os.listdir(directory):
        subject_dir = os.path.join(directory, file, 'anat')
        if not os.path.exists(subject_dir):
            continue

        # Iterate over each file in the directory
        for filename in os.listdir(subject_dir):
            for dataset_fold in dataset_folds:
                # Note: the CSV files were generated using the compute_metrics_reloaded.py script:
                #  https://github.com/ivadomed/MetricsReloaded/blob/main/compute_metrics_reloaded.py
                if filename.endswith(f'{dataset_fold}_metrics.csv') and any(
                        contrast_name in filename for contrast_name in contrast):
                    # Skip files containing "UNIT1_neg" if the filename also contains "UNIT1"
                    if "UNIT1_neg" in filename and contrast == ["UNIT1"]:
                        continue
                    else:
                        # Extract subject, dataset, and fold from the filename
                        parts = filename.split('_')
                        subject = parts[1]  # Assuming subject ID is the second part
                        if 'T2wmodel' in filename and 'dseg' in filename:
                            dataset = 'T2wmodel_dseg'
                            fold = ''
                        else:
                            dataset = parts[-3]  # Assuming dataset is the second to last part (e.g., Dataset011)
                            fold = '_'+parts[-2]  # Assuming fold is the last part (e.g., fold0)
                            contrast_name = parts[2]
                        contrast_value.append(contrast_name)

                        # Read the CSV file
                        df = pd.read_csv(os.path.join(subject_dir, filename))

                        # Add columns for subject and dataset
                        df['subject'] = subject
                        df['dataset'] = dataset+fold
                        df['contrast'] = contrast_value[0].replace("-", "").upper()
                        contrast_value = []

                        # Append dataframe to list
                        dataframes.append(df)
                        file_names.append(filename)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Make column label as only integer value
    combined_df['label'] = combined_df['label'].astype(int)

    # Extract unique dataset and fold combinations from the 'dataset' column
    unique_datasets = sorted(combined_df['dataset'].unique())

    # Convert the 'dataset' column to a categorical type with the dynamically generated order
    combined_df['dataset'] = pd.Categorical(combined_df['dataset'], categories=unique_datasets, ordered=True)

    # Sort the DataFrame by the 'dataset' column - first will be T2w model
    combined_df = combined_df.sort_values('dataset', ascending=True)

    # Rename dataset names to custom names - if Dataset contains 026 rename it to UNIT1-neg_v1, 027 to MULTICON-v1, 028 to UNIT1-neg_v2 and 028 to MULTICON-v2 and T2wmodel_dseg to T2w model (r240523); only if datasets are in unique_datasets
    if 'Dataset026_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset026_fold0', 'UNIT1-neg_v1')
    if 'Dataset027_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset027_fold0', 'MULTICON_v1')
    if 'Dataset028_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset028_fold0', 'UNIT1-neg_v2')
    if 'Dataset029_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset029_fold0', 'MULTICON_v2')
    if 'T2wmodel_dseg' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('T2wmodel_dseg', 'T2w (r20240523)')


    # Remove columns EmptyPred and EmptyRef
    combined_df = combined_df.drop(columns=['EmptyPred', 'EmptyRef'])

    # Left in the table only rows with wanted contrasts
    contrast = [contrast_name[:5].replace("-", "").upper() for contrast_name in contrast]
    combined_df = combined_df[combined_df['contrast'].isin(contrast)]

    # Save the combined dataframe to a new CSV file if the output_combined_csv argument is provided
    if output:
        contrast_check = '_'.join(contrast)  # Create a string representation of the contrast list
        unique_datasets_concat = '_'.join(unique_datasets)  # Create a string representation of the unique datasets
        combined_df.to_csv(
            f"{output}/combined-dataframe_{contrast_check}_{unique_datasets_concat}.csv",
            index=False
        )
        print(
            f'Saved the combined dataframe to {output}/combined-dataframe_{contrast_check}_{unique_datasets_concat}.csv')

    # from combined_df calculate mean and std for all levels together and distinguis between datasets to new dataframe
    mean_std_df = combined_df.groupby(['dataset'])['DiceSimilarityCoefficient'].agg(['mean', 'std'])
    mean_std_df = mean_std_df.rename(columns={'mean': 'mean_all', 'std': 'std_all'})
    mean_std_df = mean_std_df.reset_index()
    print(mean_std_df)


    return combined_df


def create_boxplot_or_violinplot(combined_df, plot_type, contrast, dataset_folds, output=''):

    """
    This function creates a boxplot or violinplot from the combined dataframe.
    :param combined_df: Combined dataframe.
    :param plot_type: Type of plot you want to create.
    :param contrast: Name of contrast, that you want to visualize.
    """

    unique_dataset_names = sorted(combined_df['dataset'].unique())

    if len(unique_dataset_names) == 3 and 'UNIT1_neg' not in contrast:
        unique_dataset_names[1], unique_dataset_names[0], unique_dataset_names[2] = unique_dataset_names[0], unique_dataset_names[2], unique_dataset_names[1]

    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    #sns.set(style="dark")
    #sns.set_theme()

    if plot_type == 'boxplot':
        if len(contrast) == 1:
            if contrast == ["inv-1_part-mag_MP2RAGE"]:
                contrast_legend = "contrast: INV1"
            elif contrast == ["inv-2_part-mag_MP2RAGE"]:
                contrast_legend = "contrast: INV2"
            elif contrast == ["UNIT1"]:
                contrast_legend = "contrast: UNIT1"
            else:
                contrast_legend = "contrast: UNIT1-neg"
                
            ax = sns.boxplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df, hue='dataset', hue_order=unique_dataset_names, dodge=True, width=0.6)
            legend = plt.legend(title='Model name', title_fontsize=18, fontsize=18, loc='upper center',
                       bbox_to_anchor=(1.23, 1.00), ncol=1, frameon=False)
            plt.title(f"    Comparison of models according to spinal levels ({contrast_legend})",
                      fontsize=23, pad=20)
            plt.xlabel('Spinal level', fontsize=20)
            plt.ylabel('Dice Similarity Coefficient', fontsize=24)
            plt.subplots_adjust(right=0.72)
            # plt.legend(title='', fontsize=15, loc='upper right')
            ax.yaxis.set_major_locator(MultipleLocator(0.10))
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
            plt.ylim(-0.01, 1.01)
            plt.tight_layout()
            if output:
                contrast_check = '_'.join(contrast)  # Combine all contrast names into a single string
                dataset_names_concat = '_'.join(dataset_folds)  # Combine all dataset names into a single string
                plt.savefig(f"{output}/{contrast_check}_plot_{dataset_names_concat}.png", bbox_extra_artists=[legend], bbox_inches='tight')


        elif len(contrast) != 1 and len(dataset_folds) == 1:
            colormap = sns.color_palette("tab10")[3:5]
            colormap.append(sns.color_palette("tab10")[9])
            ax = sns.boxplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df, hue='contrast', width=0.6, palette=colormap)
            plt.title(f'    Statistical comparison of {unique_dataset_names[0]} model on different contrasts', fontsize=23, pad=20)
            legend = plt.legend(title='Contrast', title_fontsize=18, fontsize=18, loc='upper center',
                                bbox_to_anchor=(1.12, 1.00), ncol=1, frameon=False)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('Spinal level', fontsize=20)
            plt.ylabel('Dice Similarity Coefficient', fontsize=24)
            plt.subplots_adjust(right=0.90)
            # plt.legend(title='', fontsize=15, loc='upper right')
            ax.yaxis.set_major_locator(MultipleLocator(0.10))
            plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
            plt.ylim(-0.01, 1.01)
            plt.tight_layout()

            if output:
                contrast_check = '_'.join(contrast)  # Combine all contrast names into a single string
                dataset_names_concat = '_'.join(dataset_folds)  # Combine all dataset names into a single string
                plt.savefig(f"{output}/{contrast_check}_plot_{dataset_names_concat}.png")




    elif plot_type == 'violinplot':
        sns.violinplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df, hue='dataset', hue_order=unique_dataset_names,dodge=True)
        if output:
            contrast_check = '_'.join(contrast)  # Combine all contrast names into a single string
            dataset_names_concat = '_'.join(dataset_folds)  # Combine all dataset names into a single string
            plt.savefig(f"{output}/{contrast_check}_plot_{dataset_names_concat}.png")

    plt.xlabel('Spinal level', fontsize=21)
    plt.ylabel('Dice Similarity Coefficient', fontsize=21)
    #plt.legend(title='', fontsize=15, loc='upper right')

    plt.ylim(-0.01, 1.01)

    plt.show()


def create_mean_std_table(combined_df, contrast, output=''):
    """
    This function creates a table with mean and standard deviation values for each dataset and fold.
    :param combined_df: Combined dataframe.
    :return: Table with mean and standard deviation values.
    """

    # Group the dataframe by dataset and fold and calculate the mean and standard deviation values
    agg_table = combined_df.groupby(['dataset', 'label'])['DiceSimilarityCoefficient'].agg(['mean', 'std'])

    # Combine the mean and std into a single string formatted as "mean ± std"
    mean_std_table = agg_table.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1)

    # Unstack the table to have datasets in rows and labels in columns
    mean_std_table = mean_std_table.unstack(level=1)

    # rename columns (e.g. 2 --> spinal level 2)
    mean_std_table.columns = [f"spinal level {col}" for col in mean_std_table.columns]

    # get dataset names
    dataset_names = mean_std_table.index
    dataset_names_concat = '_'.join(dataset_names)  # Combine all dataset names into a single string

    # save the table
    if output:
        contrast_concat = '_'.join(contrast)
        mean_std_table.to_csv(f'{output}/mean_std_table_{contrast_concat}_{dataset_names_concat}.csv')
    return mean_std_table


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    directory = args.i
    dataset_folds = args.dataset_folds
    analysed_contrast = args.contrast
    output = args.output
    plot_type = args.plot_type

    # Create the combined dataframe
    combined_dataframe = create_dataframe(directory, dataset_folds, output,
                                          analysed_contrast)

    # Create the table with mean and standard deviation values
    create_mean_std_table(combined_dataframe, analysed_contrast, output)

    # Create the boxplot/violinplot
    create_boxplot_or_violinplot(combined_dataframe, plot_type, analysed_contrast, dataset_folds, output)


if __name__ == "__main__":
    main()


