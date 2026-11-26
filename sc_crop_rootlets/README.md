# rootletseg-crop

Does cropping the images with **sc-crop** before training improve the rootlet segmentation (RootletSeg)?
This folder trains/tests an nnU-Net on images cropped around the spinal canal, brings its predictions back to the
original images, and compares them with the existing model on the same test data and the same ground truth.

```
original nnU-Net dataset ──► 1. crop (sc-crop) ──► cropped dataset ──► 2. train + predict (nnU-Net, elsewhere)
                                    │ cropbox/                                       │ predictions (cropped grid)
                                    ▼                                                ▼
                            3. pad back to the original grid ◄───────────────────────┘
                                    ▼
original labelsTs ──► 4. metrics (MetricsReloaded) ──► 5. compare with the existing model (same cases, same GT)
```

| Step | Script | Output |
|---|---|---|
| 1. Crop | `crop_nnunet_dataset.py` | cropped nnU-Net dataset, `cropbox/<case>_cropbox.nii.gz`, `crop_log.csv` |
| 2. Train + predict | nnU-Net (elsewhere, e.g. on the cluster) | predictions on the cropped test images |
| 3. Pad back | `pad_predictions.py` | predictions on the original grid, `pad_log.csv` |
| 4. Metrics | `sc_crop_rootlets_stats.py` | `metrics_per_case_level.csv`, `stats_*.csv`, box plots |
| 5. Compare | `compare_models.py` | `comparison_*.csv`, `figures/dice_*.png`, `figures/hausdorff_*.png` |

## Requirements

- `sc_crop` environment of this repository (`pip install -r ../requirements.txt`) and the weights `model_detect_canal_sc.pt` (see the main README)
- nnU-Net v2 (step 2)
- pandas, matplotlib, seaborn (steps 4-5; figures in the style of `dice-hd95-graph.py`, see `rootlet_graphs.py`)
- [MetricsReloaded](https://github.com/ivadomed/MetricsReloaded) in `~/PycharmProjects/MetricsReloaded` (step 4, path set in `METRICS_SCRIPT` of `sc_crop_rootlets_stats.py`)

## 1. Crop the dataset

```bash
python rootletseg-crop/crop_nnunet_dataset.py \
    --dataset $nnUNet_raw/Dataset101_Rootlets --output-dir $nnUNet_raw/Dataset102_RootletsCrop \
    --model model_detect_canal_sc.pt --structure canal --conf 0.26 0.29 --margin 10 10 30
```

Crops `imagesTr`/`labelsTr` and `imagesTs`/`labelsTs` with the same box per case. Check `crop_log.csv`:
`label_voxels_lost` > 0 means rootlets outside the box (they can never be predicted), `error` means the crop failed
(case not in the cropped dataset). Keep the `cropbox/` folder: it is needed in step 3.

## 2. Train and predict (elsewhere)

Copy `Dataset102_RootletsCrop` to the training machine, then as usual, e.g.:

```bash
nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity
nnUNetv2_train 102 3d_fullres all
nnUNetv2_predict -d 102 -c 3d_fullres -f all -i $nnUNet_raw/Dataset102_RootletsCrop/imagesTs -o predictions_crop
```

Bring `predictions_crop/` back (`<case>.nii.gz`, on the grid of the cropped images).

## 3. Pad the predictions back to the original images

```bash
python rootletseg-crop/pad_predictions.py \
    --pred-root predictions_crop --cropbox-dir $nnUNet_raw/Dataset102_RootletsCrop/cropbox \
    --output-root predictions_crop_padded/predict_fold_all \
    --reference-dir $nnUNet_raw/Dataset101_Rootlets/labelsTs --empty-for-missing
```

Writes `predictions_crop_padded/predict_fold_all/<case>.nii.gz` directly into `--output-root` (no subfolder is created). If `--pred-root` contains several
`predict_fold_*` folders, they are kept as subfolders of `--output-root`.
`--reference-dir` = the ORIGINAL labels: checks the padded predictions are on their grid and lists the test cases
without a prediction. With `--empty-for-missing` an empty prediction is written for them, so a failed crop counts as
a failure in the evaluation instead of being silently left out.

## 4. Metrics against the original ground truth (both models)

Same evaluation as the RootletSeg paper by default: Dice and HD95 (`-metrics dsc hd_perc`) for all levels, levels
empty in both GT and prediction kept. Post-processing: C1 is then removed from all statistics, tables, figures and
`metrics_per_case_level.csv` (used by `compare_models.py`), T1 is kept; all levels stay in
`metrics_per_case_level_all_levels.csv`. `-metrics dsc hd` for the maximum Hausdorff distance (per-case CSVs are
recomputed automatically when `-metrics` changes). To reproduce the paper for Dataset037, the ground truth and the
test cases must be the paper's (17 scans: 4 MP2RAGE participants x 3 contrasts + 5 T2w); expected overall Dice
0.65 ± 0.10, HD95 5.10 ± 1.82 mm (row `C2-C8 (paper overall)` of `stats_overall.csv` / `summary_overall.csv`:
the paper's overall values leave out T1, its per-contrast values keep it).

Both models must be evaluated against the SAME ORIGINAL `labelsTs` (not the cropped labels):

```bash
# sc-crop model (padded predictions)
python3 rootletseg-crop/sc_crop_rootlets_stats.py -labels $nnUNet_raw/Dataset101_Rootlets/labelsTs \
    -pred-root predictions_crop_padded/predict_fold_all -output metrics_crop
# existing model, run on the ORIGINAL imagesTs (folder named predict_fold_*, files <case>.nii.gz)
python3 rootletseg-crop/sc_crop_rootlets_stats.py -labels $nnUNet_raw/Dataset101_Rootlets/labelsTs \
    -pred-root predictions_existing/predict_fold_all -output metrics_existing
```

### Several models sharing one ground truth (`-root`)

If all the predictions sit next to one ground truth folder:

```
results_rootlets/
├── groundtruthsTs/           <case>.nii.gz   (ORIGINAL labels)
├── DatasetXX/predict_fold_all/   <case>.nii.gz
└── DatasetXY/predict_fold_0/ ... predict_fold_all/
```

```bash
python3 rootletseg-crop/sc_crop_rootlets_stats.py -root results_rootlets
# options: -datasets DatasetXX DatasetXY, -folds all, -gt-dirname groundtruthsTs, -output <folder>
# fold per dataset, e.g. original model vs padded sc-crop predictions (predict_fold_all_paded):
python3 rootletseg-crop/sc_crop_rootlets_stats.py -root results_rootlets -folds DatasetXX:all DatasetXY:all_paded
```

Every `Dataset*/predict_fold_*` is evaluated against `groundtruthsTs` into
`results_rootlets/metrics/<Dataset>/<predict_fold_XX>/` (same files as above), plus `metrics/metrics_all.csv` (all
datasets and folds, with a `dataset` column) and `metrics/summary_overall.csv` (Dice/HD mean ± SD per dataset and
fold), and `metrics/figures/dice_ALL.png` + `hausdorff_ALL.png`: all datasets side by side per spinal level, in the
style of `dice-hd95-graph.py` (+ `dice_<contrast>.png`, ... per contrast). Short legend names and their order with
`-names DATASET[:FOLD]=LABEL`, e.g. `-names Dataset037_...=RootletSeg "Dataset052_...:all_paded=RootletSeg + sc-crop"`.
For each dataset/fold, `figures/dice_hausdorff_contrasts.png` shows a) Dice and b) Hausdorff, one box per contrast
and spinal level (RootletSeg paper figure). Predictions of a cropped model must already be padded back (step 3, e.g. `--output-root
results_rootlets/DatasetXY/predict_fold_all` for one prediction folder).

## 5. Compare the models

```bash
python rootletseg-crop/compare_models.py \
    --metrics-a metrics_existing/predict_fold_all/metrics_per_case_level.csv --name-a "RootletSeg" \
    --metrics-b metrics_crop/predict_fold_all/metrics_per_case_level.csv --name-b "RootletSeg + sc-crop" \
    --output comparison
```

With the `-root` layout, give the datasets instead of the CSV paths (names default to the dataset names, output to
`metrics/comparison_DatasetXX_vs_DatasetXY`):

```bash
python rootletseg-crop/compare_models.py --metrics-root results_rootlets/metrics \
    --dataset-a DatasetXX --dataset-b DatasetXY --fold all        # --fold-b if model B uses another fold
python rootletseg-crop/compare_models.py --metrics-root results_rootlets/metrics \
    --dataset-a DatasetXX --fold all --dataset-b DatasetXY --fold-b all_paded \
    --name-a "RootletSeg" --name-b "RootletSeg + sc-crop"
```

Rows are paired by case and level (rows of only one model are listed and left out). For each metric (difference =
B - A): mean ± SD overall, per level and per contrast; paired test on the per-case means (the levels of a case are
not independent) and per level across cases (Holm-corrected). The test follows the normality of the differences
(Shapiro-Wilk): paired t-test if normal, otherwise Wilcoxon signed-rank test. Dice: higher is better; HD: lower
is better.

## Checks for a fair comparison

- Same test cases for both models, and none of them in the training set of the existing model.
- Same ground truth: the original `labelsTs` for both (step 4).
- Failed crops counted (`--empty-for-missing` in step 3), and `label_voxels_lost` in `crop_log.csv` reported.
