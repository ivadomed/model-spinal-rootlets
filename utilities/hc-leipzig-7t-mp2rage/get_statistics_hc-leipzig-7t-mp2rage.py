import argparse
import os
import re
import numpy as np

'''
This script computes metrics for nnUNetv2 results and saves them as csv files wit usage of MetricsReloader package.
It is necessary to download the MetricsReloader package from https://github.com/ivadomed/MetricsReloaded.git 

Usage: python compute_metrics_nnunetv2.py -i /path/to/data_processed -test-subjects sub-01 sub-02 sub-03 -metrics dsc 
'''


def parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Compute metrics for nnUNetv2 results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i',
        required=True,
        help='Path to the data processed folder'
    )
    parser.add_argument(
        '-test-subjects',
        required=True,
        nargs='+',
        help='List of test subjects (e.g. sub-01 sub-02 sub-03)'
    )
    parser.add_argument(
        '-metrics',
        required=False,
        nargs='+',
        help='List of metrics to compute. Default is DSC',
        default=['dsc'],
        choices=['dsc', 'hd', 'fbeta', 'nsd', 'vol_diff', 'rel_vol_error', 'lesion_ppv', 'lesion_sensitivity',
                 'lesion_f1_score', 'ref_count', 'pred_count', 'lcwa']
    )
    return parser


def compute_metrics(input_folder, subjects, metrics):
    """
    This function computes metrics from results organised in BIDS and saves them as csv files.
    :param input_folder: input folder with BIDS structured data
    :param subjects: selected subjects for which the metrics will be computed
    :param metrics: selected metrics to compute
    """

    # Regular expression to find files with predictions from exact dataset and fold
    pattern = re.compile(r'Dataset\d{3}_fold\d{1}\.nii\.gz$')

    # Convert list of metrics to string for MetricsReloader
    metrics_str = ' '.join(metrics)

    # Iterate through each subject and compute metrics
    for subject in subjects:
        subject_anat_dir = os.path.join(input_folder, subject, 'anat')
        if os.path.exists(subject_anat_dir):
            for root, dirs, files in os.walk(subject_anat_dir):
                for file in files:
                    # Find file that ends with "GT.nii.gz" - reference file
                    if file.endswith("GT.nii.gz"):
                        reference = os.path.join(root, file)

                for file in files:
                    if pattern.search(file):
                        print(os.path.join(root, file))
                        output_file = file[:-7] + '_metrics.csv'
                        os.system(f'python3 /media/xkrejc78/Transcend/NeuroPoly_internship/code/MetricsReloaded/compute_metrics_reloaded.py -reference '
                                    f'{reference} -prediction {os.path.join(root, file)} -metrics {metrics_str} -output'
                                    f' {os.path.join(root, output_file)}')


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


