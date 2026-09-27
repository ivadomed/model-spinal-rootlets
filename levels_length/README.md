# Spinal levels length

Measure the distance between spinal level midpoints along the spinal cord:

- **Cervical: C2–C8**, based on the spinal rootlets segmentation
- **Lumbar: L1–S1**, based on manually labeled spinal levels

All scripts require SCT (they use the SCT Python environment).

## Scripts

| Script | What it does |
|---|---|
| `01_run_batch_rootlets_spinal_levels.sh` | Cervical, per subject (run with `sct_run_batch`): spinal cord segmentation, PMJ detection, rootlets segmentation, and spinal levels with their distances from the PMJ. Manual labels in `derivatives/labels` are used when available. |
| `02_compute_cervical_midpoints_distance.py` | Cervical: aggregates the results across subjects, computes the C2–C8 midpoints distance and plots it. |
| `03_compute_lumbar_midpoints_distance.py` | Lumbar: computes the L1–S1 midpoints distance from labeled spinal cord segmentations and plots it. |
| `config_01_run_batch_rootlets_spinal_levels.json` | `sct_run_batch` config listing the spine-generic subjects with manual rootlets and spinal cord segmentations. |

Spinal levels for the cervical analysis are computed by
[`inter-rater_variability/02a_rootlets_to_spinal_levels.py`](../inter-rater_variability/02a_rootlets_to_spinal_levels.py),
called from the batch script.

## Cervical: C2–C8

Spinal levels are obtained from the intersection of the rootlets and the spinal cord segmentation. The midpoint of
each level is the average of the distances of its start and end from the pontomedullary junction (PMJ). The C2–C8
distance is then `distance_from_pmj_midpoint` of C8 minus `distance_from_pmj_midpoint` of C2.

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
python 02_compute_cervical_midpoints_distance.py -i <path_output>/data_processed
```

Outputs per subject (in `data_processed/<subject>/anat/`):

- `*_label-rootlets_dseg_spinal_levels.nii.gz`: spinal levels projected on the spinal cord
- `*_label-rootlets_dseg_pmj_distance.csv`: per-level start, end and midpoint distances from the PMJ

Outputs across subjects (in the folder passed to `-i`):

- `spinal_levels_spine-generic.csv`: per-level distances for all subjects
- `spinal_levels_midpoints_distance_2-8_spine-generic.csv`: C2–C8 midpoints distance (mm) per subject
- `figure_spinal_levels_midpoints_distance_2-8.png`: C2–C8 midpoints distance per subject

## Lumbar: L1–S1

Input: manually labeled spinal cord segmentations (`sub-*/*_spinallevels_dseg.nii.gz`) with one value per spinal
level: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=S1, 7=S2.

The midpoint of each level is the center of mass along the superior-inferior axis, and the total L1–S1 distance 
is the number of slices between the L1 and S1 midpoints multiplied by the slice thickness.

```bash
python 03_compute_lumbar_midpoints_distance.py -i <folder with sub-*/*_spinallevels_dseg.nii.gz>
```

Outputs (in the folder passed to `-i`):

- `lumbar_midpoints_distance_L1-S1.csv`: L1–S1 midpoints distance (mm) per subject, and the min slice, max slice and
  midpoint slice of each spinal level
- `figure_lumbar_midpoints_distance_L1-S1.png`: L1–S1 midpoints distance per subject
