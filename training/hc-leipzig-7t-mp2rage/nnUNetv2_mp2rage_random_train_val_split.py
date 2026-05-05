import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import KFold
import json
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate 5-fold CV splits (subject-based, expanded to contrasts). "
                    "JSON contains train/val only, CSV includes test."
    )
    parser.add_argument('-i', required=True, type=str,
                        help='Path to input CSV (train/test split).')
    parser.add_argument('-o', required=True, type=str,
                        help='Output directory.')
    return parser


def main():
    args = get_parser().parse_args()
    input_csv = args.i
    output_dir = args.o

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv)

    # Split train/test
    train_df = df[df['Train/Test'] == 'Train']
    test_df = df[df['Train/Test'] == 'Test']

    # Extract subjects
    train_subjects = train_df['Subject'].unique()
    test_subjects = test_df['Subject'].unique()

    train_subjects = [int(s.split('-')[1]) for s in train_subjects]
    train_subjects = np.array([f"sub-{s}" for s in train_subjects])

    test_subjects = [int(s.split('-')[1]) for s in test_subjects]
    test_subjects = np.array([f"sub-{s}" for s in test_subjects])

    # Contrasts
    contrasts = [
        "inv-1_part-mag_MP2RAGE",
        "inv-2_part-mag_MP2RAGE",
        "UNIT1"
    ]

    all_subjects = np.concatenate([train_subjects, test_subjects])

    expanded_subjects = np.array([f"{subj}_{contrast}" for subj in all_subjects for contrast in contrasts])

    # Expand test once (for CSV labeling only)
    test_expanded = [f"{subj}_{contrast}" for subj in test_subjects for contrast in contrasts]

    # KFold only on TRAIN subjects
    kf = KFold(n_splits=5, shuffle=True, random_state=7)

    folded_dataframe = pd.DataFrame({"Subject": expanded_subjects})
    fold_splits = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subjects), start=0):

        train_sub = train_subjects[train_idx]
        val_sub = train_subjects[val_idx]

        train_expanded = [f"{subject}_{contrast}" for subject in train_sub for contrast in contrasts]
        val_expanded = [f"{subject}_{contrast}" for subject in val_sub for contrast in contrasts]

        # JSON (nnUNet structure needs only train-val split without test)
        fold_splits.append({"train": train_expanded, "val": val_expanded})

        # CSV modification
        col = f"Fold_{fold}"
        folded_dataframe[col] = "Excluded"
        folded_dataframe.loc[folded_dataframe["Subject"].isin(train_expanded), col] = "Train"
        folded_dataframe.loc[folded_dataframe["Subject"].isin(val_expanded), col] = "Validation"

        # Add test info ONLY to CSV
        folded_dataframe.loc[folded_dataframe["Subject"].isin(test_expanded),col] = "Test"

    # Save JSON
    with open(os.path.join(output_dir, "MP2RAGE_fold_splits.json"), "w") as f:
        json.dump(fold_splits, f, indent=4)

    # Save CSV
    folded_dataframe.to_csv(
        os.path.join(output_dir, "MP2RAGE_fold_splits.csv"),
        index=False
    )

    print("Saved JSON (train/val only) and CSV (including test labels).")


if __name__ == "__main__":
    main()