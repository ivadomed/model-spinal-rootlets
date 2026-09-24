# Spinal levels length

Measure the distance between the **C2 and C8 spinal level midpoints** along the spinal cord, based on the spinal
rootlets segmentation.

Spinal levels are obtained from the intersection of the rootlets and the spinal cord segmentation. The midpoint of
each level is the average of the distances of its start and end from the pontomedullary junction (PMJ). The C2–C8
distance is then `distance_from_pmj_midpoint` of C8 minus `distance_from_pmj_midpoint` of C2.

## Scripts

| Script | What it does |
|---|---|
| `01_run_batch_rootlets_spinal_levels.sh` | Per-subject processing (run with `sct_run_batch`): spinal cord segmentation, PMJ detection, rootlets segmentation, and spinal levels with their distances from the PMJ. Manual labels in `derivatives/labels` are used when available. |
| `02_compute_spinal_levels_length.py` | Aggregates the results across subjects, computes the C2–C8 midpoints distance and plots it. |


## How to run

Requires SCT (the scripts use the SCT Python environment).

1. Download the data (the images are stored with git-annex):

```bash
cd ~/data/data.neuro.polymtl.ca/data-multi-subject
git annex get sub-*/anat/*_T2w.nii.gz derivatives/labels/sub-*/anat/*_T2w_label-*
```

2. Run the per-subject processing (edit `path_data` and `path_output` in the config if needed):

```bash
sct_run_batch -config config_01_run_batch_rootlets_spinal_levels.json
```

3. Aggregate the results:

```bash
python 02_compute_spinal_levels_length.py -i <path_output>/data_processed
```

## Outputs

Per subject (in `data_processed/<subject>/anat/`):

- `*_label-rootlets_dseg_spinal_levels.nii.gz`: spinal levels projected on the spinal cord
- `*_label-rootlets_dseg_pmj_distance.csv`: per-level start, end and midpoint distances from the PMJ

Across subjects (in the folder passed to `-i`):

- `spinal_levels_spine-generic.csv`: per-level distances for all subjects
- `spinal_levels_midpoints_distance_2-8_spine-generic.csv`: C2–C8 midpoints distance (mm) per subject
- `figure_spinal_levels_midpoints_distance_2-8.png`: C2–C8 midpoints distance per subject
