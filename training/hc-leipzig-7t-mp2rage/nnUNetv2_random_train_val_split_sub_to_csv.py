import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json

# Load the data
train_test_dataframe = pd.read_csv('hc-leipzig-7t-mp2rage_train-test_split.csv')

# Get the training data only
train_dataframe = train_test_dataframe[train_test_dataframe['Train/Test'] == 'Train']

# Extract unique subjects
subjects = train_dataframe['Subject'].unique()

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=7)

# Create a copy of the dataframe to add fold information
folded_dataframe = train_test_dataframe.copy()

# Prepare the list to store each fold's splits for JSON
fold_splits = []

# Add columns for each fold
for fold, (train_index, val_index) in enumerate(kf.split(subjects), start=1):
    train_subjects = subjects[train_index]
    val_subjects = subjects[val_index]

    # Save the split into the JSON-style dictionary
    fold_splits.append({
        "train": train_subjects.tolist(),
        "val": val_subjects.tolist()
    })

    # Create a new column for this fold in the dataframe
    fold_column = f'Fold_{fold}'
    folded_dataframe[fold_column] = 'Excluded'  # Initialize with 'Excluded'

    # Update the column for train and validation subjects
    folded_dataframe.loc[folded_dataframe['Subject'].isin(train_subjects), fold_column] = 'Train'
    folded_dataframe.loc[folded_dataframe['Subject'].isin(val_subjects), fold_column] = 'Validation'

# Save the dataframe with fold columns to a CSV file
folded_dataframe.to_csv('hc-leipzig-7t-mp2rage_with_folds.csv', index=False)

# Print confirmation
print("Dataframe with folds saved to 'hc-leipzig-7t-mp2rage_with_folds.csv'")
print("Fold splits saved to 'hc-leipzig-7t-mp2rage_fold_splits.json'")

# Display the first few rows of the updated dataframe
print(folded_dataframe.head())
