# Cropped cervical rootlets training

This folder contains the dataset preparation, split, orientation, review, and results-figure scripts for the cropped RPI cervical rootlets experiment.

## Results figures

`generate_results_figures.py` creates the figures intended for a GitHub results issue:

- `test_dice_by_contrast.png`
- `test_dice_by_spinal_level.png`
- `qualitative_best_median_worst.png`
- `qualitative_t1_failures.png`
- `figure_manifest.json`, recording the selected cases and canonical RAS axial slice indices

The two Dice plots only require the per-image metrics CSV:

```console
python generate_results_figures.py \
  --metrics-per-case /path/to/metrics_per_case.csv \
  --output-dir /path/to/results/figures
```

To also generate qualitative panels, run the script where the cropped NIfTI images, references, and predictions are available:

```console
RUN=/home/kuanyiw/experiments/results/cropped_rpi_2000epochs_fold0_best
TEST=/home/kuanyiw/projects/rootlets/data/canonical_test_17/cropped_rpi

python generate_results_figures.py \
  --metrics-per-case "$RUN/metrics_per_case.csv" \
  --images-dir "$TEST/imagesTs" \
  --labels-dir "$TEST/labelsTs" \
  --predictions-dir "$RUN/test17_predictions" \
  --output-dir "$RUN/figures"
```

Dependencies are Python 3.10 or newer, NumPy, pandas, Matplotlib, and NiBabel. Generated figures belong in the experiment results directory and should be uploaded to the GitHub issue; they are not source files that need to be committed under `training/`.

The quantitative Markdown tables are generated separately from `metrics_per_case.csv`. Keeping them as native Markdown rather than rendering them as images makes the values searchable and accessible.

## Test metrics

`compute_test_metrics.py` validates that each prediction and reference have the
same shape and affine, then reports binary and semantic Dice. Semantic macro
Dice is not a single multiclass overlap: for each image, the script computes a
3D Dice independently for every spinal level present in the union of prediction
and reference, then averages those per-level values. A missed reference level
therefore scores zero, while a level absent from both is excluded.

```console
python compute_test_metrics.py \
  --cases /path/to/canonical_test_17/cases.tsv \
  --gt-dir /path/to/canonical_test_17/cropped_rpi/labelsTs \
  --pred-dir /path/to/predictions \
  --out-dir /path/to/metrics
```

The command writes per-case, per-contrast, and per-level CSV files, a Markdown
summary, and a grid-validation report. Run the same command separately for the
cropped and uncropped predictions to obtain a matched comparison.
