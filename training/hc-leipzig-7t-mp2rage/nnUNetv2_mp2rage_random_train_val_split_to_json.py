import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json

# Load the data from the CSV file
train_test_dataframe = pd.read_csv('hc-leipzig-7t-mp2rage_train-test_split.csv')

# Get the training data only
train_dataframe = train_test_dataframe[train_test_dataframe['Train/Test'] == 'Train']

# Extract unique subjects and rename them to SCDATA_<number>
subjects = train_dataframe['Subject'].unique()[:]

# Get only number from the subject name
subjects = [int(subj.split('-')[1]) for subj in subjects]

# Rename the subjects
renamed_subjects = [f"SCDATA_{subj}" for subj in subjects]

# Define the possible contrasts for each subject
contrasts = ["inv-1_part-mag_MP2RAGE", "inv-2_part-mag_MP2RAGE", "UNIT1"]

# Expand subjects to include all contrasts
expanded_subjects = []
for subject in renamed_subjects:
    for contrast in contrasts:
        expanded_subjects.append(f"{subject}_{contrast}")

# Convert expanded_subjects into a numpy array for indexing
expanded_subjects = np.array(expanded_subjects)

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

# Save the splits to a JSON file
with open('hc-leipzig-7t-mp2rage_fold_splits.json', 'w') as json_file:
    json.dump(fold_splits, json_file, indent=4)

# Print confirmation message
print("Renamed and expanded fold splits saved to 'hc-leipzig-7t-mp2rage_fold_splits_expanded_renamed.json'")

# Display the first few rows of the updated dataframe
print(folded_dataframe.head())
