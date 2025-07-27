"""
This script computes segmentation metrics (Dice, ...) using the MetricsReloaded package for nnUNetv2 results and saves
them as CSV files. One CSV file is generated for each GT-prediction pair.
It is necessary to download the MetricsReloader package from https://github.com/ivadomed/MetricsReloaded.git 

Usage: python get_statistics_dsc.py -i /path/to/data_processed -test-subjects sub-01 sub-02 sub-03 -metrics dsc
"""

import argparse
import os
import re
from pathlib import Path


def parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """

    parser = argparse.ArgumentParser(description='Compute metrics for nnUNetv2 results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', required=True, help='Path to the data processed folder')
    parser.add_argument('-test-subjects', required=True, nargs='+', help='List of test subjects '
                                                                         '(e.g. sub-01 sub-02 sub-03)')
    parser.add_argument('-metrics', required=False, nargs='+', help='List of metrics to compute. '
                        'Default is DSC', default=['dsc'], choices=['dsc', 'hd'])
    return parser


def compute_metrics(input_folder, subjects, metrics):
    """
    This function computes metrics from results organised in BIDS and saves them as csv files.
    :param input_folder: input folder with BIDS structured data
    :param subjects: selected subjects for which the metrics will be computed
    :param metrics: selected metrics to compute
    """

    # Regular expression to find files with predictions from exact dataset and fold
    pattern = re.compile(
        r'(Dataset\d{3}_fold\d|Dataset\d{3}_fold\d_\d{2}_\d{2}_\d{2}|Dataset\d{3}_fold_\d_\d{2}_\d{2}_\d{2}|Dataset\d{3}_fold_all_\d{2}_\d{2}_\d{2}|T2wmodel_dseg)\.nii\.gz$')

    # Convert a list of metrics to string for MetricsReloader
    metrics_str = ' '.join(metrics)

    # Iterate over each subject and compute metrics
    for subject in subjects:
        anat_path = Path(input_folder) / subject / "anat"
        if not anat_path.exists():
            continue
        # Find all nii.gz files in the anat folder and find the reference file
        files = list(anat_path.glob("**/*.nii.gz"))
        reference = next((file for file in files if file.name.endswith("GT.nii.gz")), None)
        if not reference:
            continue

        # Find all prediction files that match the pattern and are not GT files
        predictions = [file for file in files if pattern.search(file.name) and not file.name.endswith("GT.nii.gz")]

        # Get metrics for each prediction file
        for pred in predictions:
            output_file = pred.with_name(pred.stem + "_metrics.csv")
            os.system(
                f"python3 ~/code/MetricsReloaded/compute_metrics_reloaded.py "
                f"-reference {reference} -prediction {pred} -metrics {metrics_str} -output {output_file}"
            )


def main():
    #  Parse the command line arguments
    args = parser().parse_args()
    input_folder = args.i
    subjects = args.test_subjects
    metrics = args.metrics

    # Compute metrics for selected subjects and save them as csv files
    compute_metrics(input_folder, subjects, metrics)


if __name__ == '__main__':
    main()
