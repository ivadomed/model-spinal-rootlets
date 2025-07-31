"""
This script generates a JSON file with 5-fold cross-validation splits for the MP2RAGE dataset (all MP2RAGE data for one
subject (INV1, INV2 and UNIT1 contrasts) will be solely used for training or validation).
The script takes a CSV file with train/test data as input and outputs the JSON file with the splits for nnUNet.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import KFold
import json


def get_parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate a JSON file with 5-fold cross-validation splits for the '
                                                 'MP2RAGE dataset.')
    parser.add_argument('-i', required=True, type=str, help='Path to the input CSV file with train/test '
                                                            'data.')
    parser.add_argument('-o', required=True, type=str, help='Path to the output JSON file.')
    return parser


def make_train_val_splits(expanded_subjects, renamed_subjects, contrasts):
    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=7)

    # Prepare the dataframe for CSV and a list for JSON
    folded_dataframe = pd.DataFrame({"Subject": expanded_subjects})
    fold_splits = []

    # Add columns for each fold
    for fold, (train_index, val_index) in enumerate(kf.split(renamed_subjects), start=1):
        # Get the train and validation subjects
        train_subjects = [renamed_subjects[idx] for idx in train_index]
        val_subjects = [renamed_subjects[idx] for idx in val_index]

        # Expand train and validation subjects with all contrasts
        train_expanded = [f"{subj}_{contrast}" for subj in train_subjects for contrast in contrasts]
        val_expanded = [f"{subj}_{contrast}" for subj in val_subjects for contrast in contrasts]

        # Save the split for JSON
        fold_splits.append({
            "train": train_expanded,
            "val": val_expanded
        })

        # Add a new column to the dataframe indicating fold membership
        fold_column = f'Fold_{fold}'
        folded_dataframe[fold_column] = 'Excluded'  # Initialize with 'Excluded'
        folded_dataframe.loc[folded_dataframe['Subject'].isin(train_expanded), fold_column] = 'Train'
        folded_dataframe.loc[folded_dataframe['Subject'].isin(val_expanded), fold_column] = 'Validation'
    return folded_dataframe, fold_splits

def main():
    parser = get_parser()
    args = parser.parse_args()
    input_csv = args.i
    output_json = args.o

    # Load the data from the CSV file (this file was created during the manual correction process)
    train_test_dataframe = pd.read_csv(input_csv)

    # Get the training data only
    train_dataframe = train_test_dataframe[train_test_dataframe['Train/Test'] == 'Train']

    # Extract unique subjects and rename them to sub-<number>
    subjects = train_dataframe['Subject'].unique()[:]

    # Get only number from the subject name
    subjects = [int(subj.split('-')[1]) for subj in subjects]

    # Rename the subjects
    renamed_subjects = [f"sub-{subj}" for subj in subjects]

    # Define the possible contrasts for each subject
    contrasts = ["inv-1_part-mag_MP2RAGE", "inv-2_part-mag_MP2RAGE", "UNIT1"]

    # Expand subjects to include all contrasts
    expanded_subjects = []
    for subject in renamed_subjects:
        for contrast in contrasts:
            expanded_subjects.append(f"{subject}_{contrast}")

    # Convert expanded_subjects into a numpy array for indexing
    expanded_subjects = np.array(expanded_subjects)

    # Create the train/validation splits
    folded_dataframe, fold_splits = make_train_val_splits(expanded_subjects, renamed_subjects, contrasts)

    # Save the splits to a JSON file
    with open(f'{output_json}/hc-leipzig-7t-mp2rage_fold_splits_zk.json', 'w') as json_file:
        json.dump(fold_splits, json_file, indent=4)


if __name__ == '__main__':
    main()
