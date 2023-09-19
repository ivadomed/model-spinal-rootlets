## Inter-rater variability analysis

The following BIDS-compliant structure is assumed:

```
├── derivatives
│	└── labels
│	    ├── sub-007
│	    │	└── ses-headNormal
│	    │	    └── anat
│	    │	        ├── sub-007_ses-headNormal_T2w_label-rootlet_rater1.nii.gz
│	    │	        ├── sub-007_ses-headNormal_T2w_label-rootlet_rater2.nii.gz
│	    │	        ├── sub-007_ses-headNormal_T2w_label-rootlet_rater3.nii.gz
│	    │	        └── sub-007_ses-headNormal_T2w_label-rootlet_rater4.nii.gz
│	    ├── sub-010
│	    │	└── ses-headUp
│	    │	    └── anat
│	    │	        ├── sub-010_ses-headUp_T2w_label-rootlet_rater1.nii.gz
│	    │	        ├── sub-010_ses-headUp_T2w_label-rootlet_rater2.nii.gz
│	    │	        ├── sub-010_ses-headUp_T2w_label-rootlet_rater3.nii.gz
│	    │	        └── sub-010_ses-headUp_T2w_label-rootlet_rater4.nii.gz
│	    ├── sub-amu02
│	    │	└── anat
│	    │	    ├── sub-amu02_T2w_label-rootlet_rater1.nii.gz
│	    │	    ├── sub-amu02_T2w_label-rootlet_rater2.nii.gz
│	    │	    ├── sub-amu02_T2w_label-rootlet_rater3.nii.gz
│	    │	    └── sub-amu02_T2w_label-rootlet_rater4.nii.gz
│	    ├── sub-barcelona01
│	    │	└── anat
│	    │	    ├── sub-barcelona01_T2w_label-rootlet_rater1.nii.gz
│	    │	    ├── sub-barcelona01_T2w_label-rootlet_rater2.nii.gz
│	    │	    ├── sub-barcelona01_T2w_label-rootlet_rater3.nii.gz
│	    │	    └── sub-barcelona01_T2w_label-rootlet_rater4.nii.gz
│	    └── sub-brnoUhb03
│	        └── anat
│	            ├── sub-brnoUhb03_T2w_label-rootlet_rater1.nii.gz
│	            ├── sub-brnoUhb03_T2w_label-rootlet_rater2.nii.gz
│	            ├── sub-brnoUhb03_T2w_label-rootlet_rater3.nii.gz
│	            └── sub-brnoUhb03_T2w_label-rootlet_rater4.nii.gz
├── sub-007
│	└── ses-headNormal
│	    └── anat
│	        └── sub-007_ses-headNormal_T2w.nii.gz
├── sub-010
│	└── ses-headUp
│	    └── anat
│	        └── sub-010_ses-headUp_T2w.nii.gz
├── sub-amu02
│	└── anat
│	    └── sub-amu02_T2w.nii.gz
├── sub-barcelona01
│	└── anat
│	    └── sub-barcelona01_T2w.nii.gz
├── sub-brnoUhb03
│	└── anat
│	    └── sub-brnoUhb03_T2w.nii.gz
```

0. Combine multi-class (i.e., not binary) segmentations from individual raters into a reference segmentation using the 
STAPLE algorithm:

Note: run this command for each subject separately.

```commandline
python combine_segmentations_from_different_raters.py -i  sub-001_T2w_label-rootlet_rater1.nii.gz sub-001_T2w_label-rootlet_rater2.nii.gz sub-001_T2w_label-rootlet_rater3.nii.gz sub-001_T2w_label-rootlet_rater4.nii.gz -o sub-001_T2w_label-rootlet_staple.nii.gz
```

1. Run `run_batch_inter_rater_variability.sh` across all subjects in `inter-rater_variability` folder using `sct_run_batch` wrapper:

```commandline
sct_run_batch -script ./run_batch_inter_rater_variability.sh -path-data inter-rater_variability -path-output inter-rater_variability_2023-XX-XX -jobs 5 -script-args model-spinal-rootlets/utilities/rootlets_to_spinal_levels.py
```

The script performs inter-rater variability analysis:
 - segment spinal cord
 - detect PMJ label
 - run `rootlets_to_spinal_levels.py` to project the nerve rootlets on the spinal cord segmentation to obtain spinal
levels, and compute the distance between the pontomedullary junction (PMJ) and the start and end of the spinal level. 
The `rootlets_to_spinal_levels.py` script outputs .nii.gz file with spinal levels and saves the results in CSV files.
 - run `compute_f1_and_dice.py` to compute the F1 and Dice scores between the reference and GT segmentations. 
The `compute_f1_and_dice.py` script saves the results in CSV files.

2. Run `generate_figure_inter_rater_variability.py` to generate the figure:

```commandline
python generate_figure_inter_rater_variability.py -i inter-rater_variability_2023-XX-XX/data_processed
```

