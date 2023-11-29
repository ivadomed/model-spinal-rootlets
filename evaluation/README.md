## Evaluation of the nnUNet model across different datasets

### 1. spine-generic single-subject

A single 38 y.o. male healthy subject acquired across 19 centers.

The dataset contains 3T T2w MRI data across three vendors (Siemens, Philips, GE).

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