# Evaluation of the nnUNet model across different datasets

This README describes the evaluation of the [r20240129](https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240129) model on three test sets to generate Figures 6 to 8 of https://doi.org/10.1162/imag_a_00218.

## test-set-1: `spine-generic single-subject` dataset (Fig. 6)

A single 38 y.o. male healthy subject acquired across 19 centers.

The dataset is open-access and contains 3T T2w MRI data across three vendors (Siemens, Philips, GE).

Details: [spine-generic/data-single-subject](https://github.com/spine-generic/data-single-subject)

0. Download the dataset:

```commandline
git clone https://github.com/spine-generic/data-single-subject && \
cd data-single-subject && \
git annex init
git annex get sub-*/anat/*T2w*
```

1. Run the nnUNet model on the single-subject dataset to obtain the rootlet segmentation:

```commandline
sct_run_batch -script 01a_spine-generic_single_subject-run_prediction.sh
              -path-data <DATA> 
              -path-output <DATA>_202X-XX-XX
              -jobs 5 
              -script-args "<PATH_REPO> <PATH_TO_NNUNET_MODEL> <FOLD>"
```

2. Generate a figure showing the inter-session variability across centers and spinal levels as a distance from the PMJ:

```commandline
python 01b_spine-generic_single-subject-generate_figure_inter-subject_variablity-PMJ_COV.py
      -i /path/to/data_processed
      -participants-tsv /path/to/participants.tsv
```

## test-set-2: sub-01 from the `courtois-neuromod` dataset (Fig. 7)

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
              -script-args "<PATH_REPO> <PATH_TO_NNUNET_MODEL> <FOLD>"
```

2. Generate a figure showing the inter-session variability across sessions and spinal levels as a distance from the PMJ:

```commandline
python 02b_courtois-neuromod-generate_figure_inter-subject_variablity-PMJ_COV.py
      -i /path/to/data_processed
```

## test-set-3: `marseille-rootlets` dataset (Fig. 8)

10 subjects (2 sessions) from the marseille-rootlets dataset (private).

0. Download the dataset from our internal git-annex server.

1. Run the nnUNet model on the single-subject dataset to obtain the rootlet segmentation:

```commandline
sct_run_batch -script 03a_marseille-rootlets-run_prediction.sh
              -path-data <DATA> 
              -path-output <DATA>_202X-XX-XX
              -jobs 5 
              -include-list sub-02 sub-12 sub-13 sub-14
              -script-args "<PATH_REPO> <PATH_TO_NNUNET_MODEL> <FOLD>"
```

2. Generate a figure showing the inter-session variability across subjects and spinal levels as a distance from the PMJ:

Note that `sub-03`, `sub-04`,`sub-05`, `sub-07`, `sub-08` are skipped because PMJ is not visible on the T2w images.
`sub-06` has spinal canal stenosis and is also skipped.

```commandline
python 03b_marseille-rootlets-generate_figure_inter-subject_variablity-PMJ_COV.py
      -i /path/to/data_processed
```
