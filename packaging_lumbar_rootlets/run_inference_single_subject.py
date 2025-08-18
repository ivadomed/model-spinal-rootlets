"""
This script is used to run inference on a single subject using a nnUNetV2 model.

Note: conda environment with nnUNetV2 is required to run this script.
For details how to install nnUNetV2, see:
https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md#installation

Author: Jan Valosek

Example:
    python run_inference_single_subject.py
        -i sub-001_T2w.nii.gz
        -o sub-001_T2w_label-rootlet.nii.gz
        -path-model <PATH_TO_MODEL_FOLDER>
        -tile-step-size 0.5
        -fold 1
"""


import os
import shutil
import subprocess
import argparse
import datetime

import torch
import glob
import time
import tempfile

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment an image using nnUNet model.')
    parser.add_argument('-i', help='Input image to segment. Example: sub-001_T2w.nii.gz', required=True)
    parser.add_argument('-o', help='Output filename. Example: sub-001_T2w_label-rootlet.nii.gz', required=True)
    parser.add_argument('-path-model', help='Path to the model folder. This folder should contain individual '
                                            'folders like fold_0, fold_1, etc. and dataset.json, '
                                            'dataset_fingerprint.json and plans.json files.', required=True, type=str)
    parser.add_argument('-use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('-fold', type=str, required=True,
                        help='Fold(s) to use for inference. Example(s): 2 (single fold), 2,3 (multiple folds), '
                             'all (fold_all).', choices=['0', '1', '2', '3', '4', 'all'])
    parser.add_argument('-use-best-checkpoint', action='store_true', default=False,
                        help='Use the best checkpoint (instead of the final checkpoint) for prediction. '
                             'NOTE: nnUNet by default uses the final checkpoint. Default: False')
    parser.add_argument('-tile-step-size', default=0.5, type=float,
                        help='Tile step size defining the overlap between images patches during inference. '
                             'Default: 0.5 '
                             'NOTE: changing it from 0.5 to 0.9 makes inference faster but there is a small drop in '
                             'performance.')

    return parser


def get_orientation(file):
    """
    Get the original orientation of an image
    :param file: path to the image
    :return: orig_orientation: original orientation of the image, e.g. LPI
    """

    # Fetch the original orientation from the output of sct_image
    sct_command = "sct_image -i {} -header | grep -E qform_[xyz] | awk '{{printf \"%s\", substr($2, 1, 1)}}'".format(
        file)
    orig_orientation = subprocess.check_output(sct_command, shell=True).decode('utf-8')
    return orig_orientation


def tmp_create():
    """
    Create temporary folder and return its path
    """
    prefix = f"sciseg_prediction_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    print(f"Creating temporary folder ({tmpdir})")
    return tmpdir


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.
    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    Taken (shamelessly) from: https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    # If no special case, behaves like the regular splitext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension. Taken (shamelessly) from:
    https://github.com/spinalcordtoolbox/manual-correction/blob/main/utils.py
    :param fname: absolute or relative file name. Example: t2.nii.gz
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii
    Examples:
    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def main():
    parser = get_parser()
    args = parser.parse_args()

    fname_file = os.path.expanduser(args.i)
    fname_file_out = os.path.expanduser(args.o)
    print(f'\nFound {fname_file} file.')

    # If the fname_file is .nii, gzip it
    # This is needed, because the filename suffix must match the `file_ending` in `dataset.json`.
    # Context: https://github.com/ivadomed/model-spinal-rootlets/issues/49
    if not fname_file.endswith('.nii.gz'):
        print('Compressing the input image...')
        os.system('gzip -f {}'.format(fname_file))
        fname_file = fname_file + '.gz'
        print(f'Compressed {fname_file}')

    # Add .gz suffix to the output file if not already present. This is needed because we gzip the input file.
    if not fname_file_out.endswith('.gz'):
        fname_file_out = fname_file_out + '.gz'

    # Create temporary directory in the temp to store the reoriented images
    tmpdir = tmp_create()
    # Copy the file to the temporary directory using shutil.copyfile
    # NOTE: Add the `_0000` suffix, because nnUNet removes the last five characters:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py#L171C19-L172C51
    # Context: https://github.com/ivadomed/model-spinal-rootlets/issues/49
    fname_file_tmp = os.path.join(tmpdir, os.path.basename(add_suffix(fname_file, '_0000')))
    shutil.copyfile(fname_file, fname_file_tmp)
    print(f'Copied {fname_file} to {fname_file_tmp}')

    # Get the original orientation of the image, for example LPI
    orig_orientation = get_orientation(fname_file_tmp)

    # Reorient the image to LPI orientation if not already in LPI
    if orig_orientation != 'LPI':
        print(f'Original orientation: {orig_orientation}')
        print(f'Reorienting to LPI orientation...')
        # reorient the image to LPI using SCT
        os.system('sct_image -i {} -setorient LPI -o {}'.format(fname_file_tmp, fname_file_tmp))

    # Note: even a single file must be in a list of lists
    fname_file_tmp_list = [[fname_file_tmp]]

    # Use fold_all (all train/val subjects were used for training) or specific fold(s)
    folds_avail = 'all' if args.fold == 'all' else [int(f) for f in args.fold.split(',')]
    print(f'Using fold(s): {folds_avail}')

    # Create directory for nnUNet prediction
    tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
    fname_prediction = os.path.join(tmpdir_nnunet, os.path.basename(add_suffix(fname_file_tmp, '_pred')))
    os.mkdir(tmpdir_nnunet)

    # Run nnUNet prediction
    print('Starting inference...it may take a few minutes...\n')
    start = time.time()
    # directly call the predict function
    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,     # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,      # applies gaussian noise and gaussian blur
        use_mirroring=False,    # test time augmentation by mirroring on all axes
        perform_everything_on_device=True,
        #device=torch.device('cuda') if args.use_gpu else torch.device('cpu'),
        device=torch.device('mps'),
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    print('Running inference on device: {}'.format(predictor.device))

    # args.path_model can contain either 'checkpoint_latest.pth' or 'checkpoint_final.pth' (depending on the nnUNet
    # version)
    checkpoint_name = 'checkpoint_final.pth' if (
        os.path.isfile(os.path.join(os.path.expanduser(args.path_model),
                                    f'fold_{folds_avail[0]}',
                                    'checkpoint_final.pth'))) else 'checkpoint_latest.pth'
    # use 'checkpoint_best.pth' if 'args.use_best_checkpoint' is True
    if args.use_best_checkpoint:
        checkpoint_name = 'checkpoint_best.pth'

    print(f'Using checkpoint: {checkpoint_name}')

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(os.path.expanduser(args.path_model)),
        use_folds=folds_avail,
        checkpoint_name=checkpoint_name,
    )
    print('Model loaded successfully. Fetching data...')

    # NOTE: for individual files, the image should be in a list of lists
    predictor.predict_from_files(
        list_of_lists_or_source_folder=fname_file_tmp_list,
        output_folder_or_list_of_truncated_output_files=tmpdir_nnunet,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    end = time.time()

    print('Inference done.')
    total_time = end - start
    print('Total inference time: {} minute(s) {} seconds\n'.format(int(total_time // 60), int(round(total_time % 60))))

    # Copy .nii.gz file from tmpdir_nnunet to tmpdir
    pred_file = glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz'))[0]
    shutil.copyfile(pred_file, fname_prediction)
    print(f'Copied {pred_file} to {fname_prediction}')

    # Reorient the image back to original orientation
    # skip if already in LPI
    if orig_orientation != 'LPI':
        print(f'Reorienting to original orientation {orig_orientation}...')
        # reorient the image to the original orientation using SCT
        os.system('sct_image -i {} -setorient {} -o {}'.format(fname_prediction, orig_orientation, fname_prediction))

    # Copy level-specific (i.e., non-binary) segmentation
    shutil.copyfile(fname_prediction, fname_file_out)
    print(f'Copied {fname_prediction} to {fname_file_out}')

    print('Deleting the temporary folder...')
    # Delete the temporary folder
    shutil.rmtree(tmpdir)

    print('-' * 50)
    print(f"Input file: {fname_file}")
    print(f"Rootlet segmentation: {fname_file_out}")
    print('-' * 50)


if __name__ == '__main__':
    main()