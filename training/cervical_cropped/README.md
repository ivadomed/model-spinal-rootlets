# Cropped cervical rootlets training

This folder contains the dataset preparation, split, orientation, review, and results-figure scripts for the cropped RPI cervical rootlets experiment.

## Reviewed C2--T1 label handoff

`prepare_handoff_dataset.py` is the preparation entry point for the reviewed
`Rootlets-train-test-labels` package. The package contains labels only, so the
script pairs its 93 split-CSV cases with source images, proves voxel-lattice
equivalence, applies lossless orientation fixes where necessary, and writes a
new derived dataset without changing the handoff or source images.

The crop policy uses the sc-crop 0.6.0 defaults plus at least 10 mm on every
anatomical face. A detector audit of all 93 split cases found that anterior
padding of 25 mm still missed one label by 5.46 mm, so the fixed configuration
is 50 mm superior, 110 mm inferior, 25 mm left/right, 40 mm anterior, and 32 mm
posterior. This fixed policy replaces per-case label-derived expansion. Labels
are consulted only after detection to verify that every foreground voxel was
retained; the command stops on a failure.

```console
python training/cervical_cropped/prepare_handoff_dataset.py \
  --data-root /home/kuanyiw/projects/rootlets/data \
  --labels-root /home/kuanyiw/projects/rootlets/data/Rootlets-train-test-labels \
  --csv training/cervical_cropped/MP2RAGE_T2w_fold_splits.csv \
  --output /home/kuanyiw/projects/rootlets/data/Dataset403_CervicalRootletsCroppedCleanT1RPI \
  --device cuda
```

The output has 76 cases in `imagesTr`/`labelsTr` and the held-out 17 in
`imagesTs`/`labelsTs`. `dataset_manifest.json` records every source path,
orientation action, padding value, bounding box, input-label checksum, and crop
QC result. The nine additional longitudinal labels in the handoff are recorded
as inventory extras but are not silently added to the published split.

To select multiple folds with `run_training.sh`, set the `FOLDS` environment
variable, for example `FOLDS="0 4"` or `FOLDS="1 all"`. Run concurrent folds
in separate GPU slots and use separate preprocessed/results directories for
this dataset.

## Training curves

Per-class validation pseudo-Dice curves can be refreshed from any nnU-Net
training log with:

```console
python training/cervical_cropped/plot_nnunet_training_log.py \
  -i /path/to/training_log_YYYY_M_D_H_M_S.txt
```

This is adapted from
`ivadomed/utilities/training_scripts/plot_nnunet_training_log.py` to support
current `np.float32(...)` log values and `fold_all`. nnU-Net also writes its
native `progress.png` in each fold result directory.

On Romane, `watch_training_curves_romane.sh` refreshes every available curve
every 30 minutes and exits after folds 0--4 and `all` have final checkpoints.

## Early stopping

Dataset403 uses `nnUNetTrainer_2000epochsEarlyStopping`, which keeps 2,000
epochs as a hard maximum but stops after all of the following are true:

- at least 1,000 epochs completed;
- EMA foreground pseudo-Dice has not improved by more than 0.002;
- that material plateau has lasted 400 epochs.

The evidence, retrospective replay, exact rule, metric interpretation, and
limitations are documented in
[`EARLY_STOPPING.md`](EARLY_STOPPING.md).

The actual best checkpoint still uses nnU-Net's unthresholded EMA comparison,
so small gains remain eligible for `checkpoint_best.pth`. The conservative
stopping thresholds preserve all 2,000 epochs in a retrospective replay of the
released uncropped `fold_all` history (best at epoch 1,998), while the earlier
cropped fold-0 history would stop near epoch 1,445 after its last material gain.

Install the repository-owned trainer into the pinned Romane environment once:

```console
bash training/cervical_cropped/install_early_stopping_trainer_romane.sh
```

If the reviewed commit is deployed beside the main Romane checkout, point the
installation, launch, and curve-watching scripts at that clean tree without
modifying the existing checkout:

```console
export ROOTLETS_REPO=/home/kuanyiw/projects/rootlets/model-spinal-rootlets-earlystop-pr108
bash "$ROOTLETS_REPO/training/cervical_cropped/install_early_stopping_trainer_romane.sh"
```

For folds 0--4, pseudo-Dice is computed from held-out validation cases. For
`fold_all`, nnU-Net uses all cases for both training and pseudo-validation, so
its stopping signal indicates optimization convergence rather than held-out
generalization. Final model comparisons must still use the 17-case test set.

The policy can be replayed against an existing checkpoint without training:

```console
python training/cervical_cropped/analyze_early_stopping_checkpoint.py \
  /path/to/checkpoint_final.pth
```

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
