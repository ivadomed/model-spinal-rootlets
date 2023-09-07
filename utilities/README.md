## Inter-rater variability analysis

1. Run `run_batch_inter_rater_variability.sh` across all subjects in `inter-rater_variability` folder using `sct_run_batch` wrapper:

```commandline
sct_run_batch -script ./run_batch_inter_rater_variability.sh -path-data inter-rater_variability -path-output inter-rater_variability_2023-09-06 -jobs 5 -script-args model-spinal-rootlets/utilities/rootlets_to_spinal_levels.py
```

The script performs inter-rater variability analysis:
- segment spinal cord
- detect PMJ label
- run `rootlets_to_spinal_levels.py` to project the nerve rootlets on the spinal cord segmentation to obtain spinal
levels, and compute the distance between the pontomedullary junction (PMJ) and the start and end of the spinal level

2. Run `generate_figure_inter_rater_variability.py` to generate the figure:

```commandline
python generate_figure_inter_rater_variability.py -i inter-rater_variability_2023-09-06/data_processed
```