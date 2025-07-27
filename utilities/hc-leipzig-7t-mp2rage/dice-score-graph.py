"""
The script reads the CSV files with Dice coefficient (and other segmentation metrics) computed using
[MetricReloaded](https://github.com/ivadomed/MetricsReloaded/blob/main/compute_metrics_reloaded.py) from the BIDS
structured data folder, creates a combined dataframe, and then creates a boxplot from the data.
The segmentation metrics can be computed by the script get_statistics_hc-leipzig-7t-mp2rage.py.

Usage for comparison of different models (one contrast):
python dice-score-graph.py -i /path/to/BIDS_structured_data -dataset-folds Dataset001_fold0 Dataset002_fold0
-contrast UNIT1 -output /path/to/output_folder

Usage for comparison of different contrasts (one model):
python dice-score-graph.py -i /path/to/BIDS_structured_data -dataset-folds Dataset001_fold0 -contrast UNIT1
inv-1_part-mag_MP2RAGE inv2_part-mag_MP2RAGE -output /path/to/output_folder

Usage for cross-validation (MULTICON model - it is neccessary to specify all folds):
python dice-score-graph.py -i /path/to/BIDS_structured_data -dataset-folds fold_0 fold_1 fold_2 fold_3 fold_4 fold_all
-contrast UNIT1 inv-1_part-mag_MP2RAGE inv2_part-mag_MP2RAGE T2w -output /path/to/output_folder

"""

import os
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch


def get_parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """

    parser = argparse.ArgumentParser(description='The script makes boxplots/violinplots from data structured in BIDS.',
                                     formatter_class=RawTextHelpFormatter, prog=os.path.basename(__file__))
    parser.add_argument('-i', required=True, help='Path to the BIDS structured data folder.')
    parser.add_argument('-dataset-folds', required=True, nargs='+', help='Dataset and fold names - e.g. '
                                                                         'Dataset001_fold0, Dataset002_fold0.')

    parser.add_argument('-contrast', required=False, help='Name of contrast, that you want to visualize.',
                        choices=['UNIT1', 'UNIT1_neg', 'inv-1_part-mag_MP2RAGE', 'inv-1_part-mag_MP2RAGE_neg',
                                 'inv-2_part-mag_MP2RAGE', 'inv-2_part-mag_MP2RAGE_neg', 'T2w'], nargs='+')

    parser.add_argument('-output', required=False, help='Path to the output folder, where you want to save '
                                                        'the combined CSV file and images')
    return parser


def create_dataframe(directory, dataset_folds, output, contrast):
    """
    This function reads the CSV files with statistical data from the BIDS structured data folder, creates a combined
    dataframe and returns it.
    :param directory: Path to the BIDS structured data folder.
    :param dataset_folds: Dataset and fold names - e.g. Dataset001_fold0, Dataset002_fold0.
    :param output: Path to the output folder, where you want to save the combined CSV file.
    :param contrast: Name of contrast(s), that you want to visualize.
    :return: Combined dataframe.
    """

    dataframes = []
    file_names = []

    # Create a list of valid filenames based on the dataset folds
    valid_filenames = ([f'{dataset_fold}_25_03_15_metrics.csv' for dataset_fold in dataset_folds] +
                       [f'{dataset_fold}_metrics.csv' for dataset_fold in dataset_folds])

    # Iterate over each subject directory (in the BIDS structured data folder)
    for file in os.listdir(directory):
        subject_dir = os.path.join(directory, file, 'anat')
        if not os.path.exists(subject_dir):
            continue

        # Iterate over each file in the subject directory
        for filename in os.listdir(subject_dir):
            # Check if the filename matches
            if not any(contrast_name in filename for contrast_name in contrast):
                continue

            if not any(filename.endswith(valid_filename) for valid_filename in valid_filenames):
                continue

            # Skip UNIT1_neg data when contrast is UNIT1 (to not bias the results)
            if "UNIT1_neg" in filename and contrast == ["UNIT1"]:
                continue

            elif contrast == ["T2w"] and (
                    "UNIT1_neg" in filename or "UNIT1" in filename or
                    "inv-1_part-mag_MP2RAGE" in filename or "inv-2_part-mag_MP2RAGE" in filename):
                continue

            # Extract subject, dataset, fold, and contrast information from filename
            parts = filename.split('_')
            subject = parts[0]
            if 'T2wmodel' in filename and 'dseg' in filename:
                dataset, fold, contrast_name = 'T2wmodel_dseg', '', 'unit1'
            elif contrast == ["UNIT1_neg"]:
                dataset = parts[-3]
                fold = '_' + parts[-2]
                contrast_name = 'UNIT1_neg'
            else:
                dataset = parts[-6]
                fold = '_' + parts[-5]
                contrast_name = parts[1]
                if parts[-4] == 'all':
                    fold = '_fold_all'
                    dataset = 'Dataset037'
                    contrast_name = parts[1]
                if parts[2] == 'T2w' or parts[3] == 'T2w':
                    contrast_name = 'T2w'

            # Read the CSV and write the subject, dataset, fold, and contrast to the dataframe
            df = pd.read_csv(os.path.join(subject_dir, filename))
            df['subject'] = subject
            df['dataset'] = dataset + fold
            df['contrast'] = contrast_name.replace("-", "").upper()

            # Append the dataframe and filename
            dataframes.append(df)
            file_names.append(filename)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract unique dataset and fold combinations from the 'dataset' column
    unique_datasets = sorted(combined_df['dataset'].unique())

    # Rename dataset names to shorter names (UNIT1-neg_v1, UNIT1-neg_v2, etc.)
    if 'Dataset026_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset026_fold0', 'UNIT1-neg_v1')
    if 'Dataset028_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset028_fold0', 'UNIT1-neg_v2')
    if 'T2wmodel_dseg' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('T2wmodel_dseg', 'T2w (r20240523)')
    if 'Dataset032_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset032_fold0', 'MP2RAGE_v2')
    if 'Dataset033_fold0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset033_fold0', 'MP2RAGE_v1')
    if 'Dataset037_fold_all' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('Dataset037_fold_all', 'MULTICON')

    # for cross-validation - model MULTICON
    if 'fold_0' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('fold_0', 'fold 0')
    if 'fold_1' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('fold_1', 'fold 1')
    if 'fold_2' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('fold_2', 'fold 2')
    if 'fold_3' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('fold_3', 'fold 3')
    if 'fold_4' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('fold_4', 'fold 4')
    if 'fold_all' in unique_datasets:
        combined_df['dataset'] = combined_df['dataset'].replace('fold_all', 'MULTICON')

    # Remove columns EmptyPred and EmptyRef
    combined_df = combined_df.drop(columns=['EmptyPred', 'EmptyRef'])

    # Get the contrast name from the command line arguments
    if "UNIT1_neg" in contrast:
        contrast = ['UNIT1_NEG']
    else:
        contrast = [contrast_name[:5].replace("-", "").upper() for contrast_name in contrast]
    combined_df = combined_df[combined_df['contrast'].isin(contrast)]

    # for UNIT1-neg and MP2RAGE models remove label 9 (the models were not trained at this level)
    if combined_df['dataset'].isin(['UNIT1-neg_v1', 'UNIT1-neg_v2', 'MP2RAGE_v1', 'MP2RAGE_v2']).any():
        combined_df = combined_df[combined_df['label'] != 9]

    # Save the combined dataframe to a new CSV file
    if output:
        contrast_names = '_'.join(contrast)
        unique_datasets_concat = '_'.join(unique_datasets)
        combined_df.to_csv(f"{output}/combined-dataframe_{contrast_names}_{unique_datasets_concat}.csv", index=False)
        print(
            f'Saved the combined dataframe to {output}/combined-dataframe_{contrast_names}_{unique_datasets_concat}.csv')

    # Calculate mean and std for all levels together and distinguish between datasets
    mean_std_df = combined_df.groupby(['dataset'])['DiceSimilarityCoefficient'].agg(['mean', 'std'])
    mean_std_df = mean_std_df.rename(columns={'mean': 'mean_all', 'std': 'std_all'})
    mean_std_df = mean_std_df.reset_index()
    print(mean_std_df)

    return combined_df, contrast


def plot_one_contrast_more_models(combined_df, unique_dataset_names, contrast):
    """
    This function creates a boxplot for different models (one contrast specified).
    :param combined_df: Combined dataframe.
    :param unique_dataset_names: Unique dataset names.
    :param contrast: Name of contrast that you want to visualize.
    :return: None
    """

    # set the color palette for the boxplot
    if 'UNIT1-neg_v1' in unique_dataset_names or 'UNIT1-neg_v2' in unique_dataset_names:
        colormap = sns.color_palette("tab20c")[17], "#87BC45", sns.color_palette("Set1")[3], sns.color_palette("Set1")[3]
        #colormap = sns.color_palette("tab20c")[17], sns.color_palette("Set1")[4], "#4CC9F0", sns.color_palette("Set1")[3]

    elif 'MP2RAGE_v1' in unique_dataset_names or 'MP2RAGE_v2' in unique_dataset_names:
        colormap = sns.color_palette("tab20c")[17], sns.color_palette("Set1")[4], "#4CC9F0", sns.color_palette("Set1")[3]

    elif 'MULTICON' in unique_dataset_names:
        colormap = "#4CC9F0", sns.color_palette("Dark2")[3], sns.color_palette("tab20c")[17]

    else:
        colormap = sns.color_palette("tab20c")
        # colormap = sns.color_palette("tab20c")[17], sns.color_palette("Dark2")[3]

    ax = sns.boxplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df,
                     hue='dataset', hue_order=unique_dataset_names, dodge=True, width=0.6, palette=colormap)

    # make custom legend names for each boxplot type
    handles, labels = ax.get_legend_handles_labels()
    if 'MULTICON' not in unique_dataset_names:
        legend = plt.legend(title='Model', title_fontsize=18, fontsize=18, loc='upper center',
                            bbox_to_anchor=(1.20, 1.00), ncol=1, frameon=False)
    else:
        new_labels = ["MP2RAGE_v2", "MULTICON", "T2w (r20240523)"]
        legend = plt.legend(handles, new_labels, title='Model', title_fontsize=24, fontsize=24, loc='upper center',
                            bbox_to_anchor=(1.35, 1.00), ncol=1, frameon=False)
    custom_patch = Patch(facecolor=sns.color_palette("tab20c")[17], edgecolor="#545455",
                         label='My Custom Label')
    handles.append(custom_patch)

    # set the title and labels
    plt.title(f"Contrast {contrast[0]}", fontsize=24, pad=10)
    plt.xlabel('Spinal level', fontsize=22)
    plt.ylabel('Dice Similarity Coefficient', fontsize=22)
    plt.subplots_adjust(right=0.70)
    # plt.legend(title='', fontsize=15, loc='upper right')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.ylim(-0.01, 1.00)
    plt.tight_layout()


def plot_one_model_more_contrasts(combined_df):
    """
    This function creates a boxplot for one model across different contrasts.
    :param combined_df: Combined dataframe.
    :return: None
    """
    # customize the color palette for the boxplot
    colormap = sns.color_palette("tab10")[3:5]
    colormap.append(sns.color_palette("tab10")[9])
    colormap.append(sns.color_palette("tab10")[8])

    # set hue order for the boxplot
    hue_order = ['INV1', 'INV2', 'UNIT1', 'T2w']

    # create the boxplot
    ax = sns.boxplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df,
                     hue='contrast', hue_order=hue_order, width=0.6, palette=colormap)
    # plt.title(f'Dice across contrasts', fontsize=23, pad=20)

    # set the parameters for the legend and labels
    plt.legend(title='Contrast', title_fontsize=18, fontsize=18, loc='upper center',
               bbox_to_anchor=(1.15, 1.00), ncol=1, frameon=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Spinal level', fontsize=22)
    plt.ylabel('Dice Similarity Coefficient', fontsize=22)
    plt.subplots_adjust(right=0.77)
    # plt.legend(title='', fontsize=15, loc='upper right')
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.ylim(-0.01, 1.00)


def plot_cross_validation(combined_df):
    """
    This function creates a boxplot for cross-validation (MULTICON model) - without specifying contrast.
    :param combined_df: Combined dataframe.
    :return: None
    """
    # set hue order for the boxplot
    hue_order = ['fold 0', 'fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold all']

    # create the boxplot
    ax = sns.boxplot(x=combined_df['label'], y=combined_df['DiceSimilarityCoefficient'], data=combined_df,
                     hue='dataset', hue_order=hue_order, dodge=True, width=0.6, palette='Set2')

    # set the parameters for the legend and labels
    # plt.title(f'    Cross-validation comparison of multi-contrast model', fontsize=23, pad=20)
    plt.legend(title='Model', title_fontsize=18, fontsize=18, loc='upper center',
               bbox_to_anchor=(1.15, 1.00), ncol=1, frameon=False)
    # plt.legend(title='', fontsize=15, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Spinal level', fontsize=22)
    plt.ylabel('Dice Similarity Coefficient', fontsize=22)
    plt.subplots_adjust(right=0.75)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.ylim(-0.01, 1.00)


def create_boxplot(combined_df, contrast, dataset_folds, output=''):
    """
    This function creates a boxplot or violinplot from the combined dataframe.
    :param combined_df: Combined dataframe.
    :param contrast: Name of contrast, that you want to visualize.
    :param dataset_folds: Dataset and fold names
    :param output: Path to the output folder, where you want to save the images.
    """

    # get unique dataset names
    unique_dataset_names = sorted(combined_df['dataset'].unique())

    # make 'T2w (r20240523)' the first dataset in the list
    if 'T2w (r20240523)' in unique_dataset_names:
        unique_dataset_names.insert(0, unique_dataset_names.pop(unique_dataset_names.index('T2w (r20240523)')))

    elif len(unique_dataset_names) == 3 and 'UNIT1_neg' not in contrast:
        unique_dataset_names[1], unique_dataset_names[0], unique_dataset_names[2] = unique_dataset_names[0], \
        unique_dataset_names[2], unique_dataset_names[1]

    # start the plot
    plt.figure(figsize=(12, 6))

    # rename combined_df['label'] values to spinal level names (to be consistent with anatomical levels)
    combined_df['label'] = combined_df['label'].replace(
        {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'C8', 9: 'T1'})

    # rename T2W to T2w in the contrast column
    combined_df['contrast'] = combined_df['contrast'].replace('T2W', 'T2w')

    # boxplot for different models (one contrast specified)
    if len(contrast) == 1:
        plot_one_contrast_more_models(combined_df, unique_dataset_names, contrast)

    # boxplot for one model across different contrasts
    elif len(contrast) != 1 and len(dataset_folds) == 1:
        plot_one_model_more_contrasts(combined_df)

    # boxplot for cross-validation (MULTICON model) - without specifying contrast
    elif len(contrast) != 1 and len(dataset_folds) != 1:
        plot_cross_validation(combined_df)

    # save the plot if output is specified
    if output:
        contrast_names = '_'.join(contrast)
        dataset_names_concat = '_'.join(dataset_folds)
        plt.savefig(f"{output}/{contrast_names}_plot_{dataset_names_concat}.png")

    plt.ylim(-0.01, 1.0)
    plt.xlabel('Spinal level', fontsize=25)
    plt.ylabel('Dice Similarity Coefficient', fontsize=25)
    plt.tight_layout()
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

    # get dataset names and combine them into a single string
    dataset_names = mean_std_table.index
    dataset_names_concat = '_'.join(dataset_names)

    # save the table if output is specified
    if output:
        contrast_concat = '_'.join(contrast)
        mean_std_table.to_csv(f'{output}/mean_std_table_{contrast_concat}_{dataset_names_concat}.csv')
    return mean_std_table


def main():
    # Get the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    directory = args.i
    dataset_folds = args.dataset_folds
    analysed_contrast = args.contrast
    output = args.output

    # Create the combined dataframe
    combined_dataframe, contrast = create_dataframe(directory, dataset_folds, output, analysed_contrast)

    # Create the table with mean and standard deviation values
    create_mean_std_table(combined_dataframe, analysed_contrast, output)

    # Create the boxplot
    create_boxplot(combined_dataframe, contrast, dataset_folds, output)


if __name__ == "__main__":
    main()
