## Evaluation of the nnUNet model across different datasets

### 1. spine-generic single-subject

A single 38 y.o. male healthy subject acquired across 19 centers.

The dataset is open-access and contains 3T T2w MRI data across three vendors (Siemens, Philips, GE).

Details: [spine-generic/data-single-subject](https://github.com/spine-generic/data-single-subject)

1. Run the nnUNet model on the single-subject dataset to obtain the rootlet segmentation:

```commandline
sct_run_batch -script 01a_spine-generic_single_subject-run_prediction.sh
              -path-data <DATA> 
              -path-output <DATA>_202X-XX-XX
              -jobs 5 
              -script-args "<PATH_TO_PYTHON_SCRIPTS> <PATH_TO_NNUNET_SCRIPT> <PATH_TO_NNUNET_MODEL>"
```

2. Generate a figure showing the inter-subject variability across centers and spinal levels as a distance from the PMJ:

```commandline
python 01b_spine-generic_single-subject-generate_figure_inter-subject_variablity-PMJ_COV.py
      -i /path/to/data_processed
      -participants-tsv /path/to/participants.tsv
```

### 2. sub-01 from the courtois-neuromod dataset

A single subject (`sub-01`, 46 y.o., male) from the open-access [courtois-neuromod/anat](https://github.com/courtois-neuromod/anat) 
dataset.

0. Download the `sub-01`:

```commandline
git clone git@github.com:courtois-neuromod/anat.git
git annex get ses-0*/anat/sub-01_ses-0*_bp-cspine_T2w.nii.gz
```

1. Run the nnUNet model on the `sub-01` to obtain the rootlet segmentation:

```commandline
sct_run_batch -script 02a_courtois-neuromod-run_prediction.sh
              -path-data <DATA> 
              -path-output <DATA>_202X-XX-XX
              -jobs 5
              -include sub-01
              -script-args "<PATH_TO_PYTHON_SCRIPTS> <PATH_TO_NNUNET_SCRIPT> <PATH_TO_NNUNET_MODEL>"
```

2. Generate a figure showing the inter-session variability across sessions and spinal levels as a distance from the PMJ:

```commandline
python 02b_courtois-neuromod-generate_figure_inter-subject_variablity-PMJ_COV.py
      -i /path/to/data_processed
```