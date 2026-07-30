# Dataset403 cropped rootlets results

## Outcome

- Five-fold out-of-fold macro level Dice: `0.618 ± 0.087`.
- Held-out macro level Dice: cropped `0.614 ± 0.064`; released `0.635 ± 0.061`.
- Paired cropped-minus-released difference: macro `−0.021 ± 0.016`; binary `−0.019 ± 0.015`.
- End-to-end time for 17 cases: cropped `1038.25 s`; released `1147.23 s` (`1.10×` faster).
- Conclusion: cropping produced a small end-to-end speed gain and lower accuracy.

## Data

- Clean handoff: `Rootlets-train-test-labels`.
- Handoff SHA-256: `4e8c0891f46e194436340eae78b8a2ae9b870968ca75acea9bc389f6928366d1`.
- Published cases: 76 train/validation + 17 test.
- Excluded: 9 inventory-only longitudinal images.
- Final target: dorsal and ventral C2–T1 rootlets.
- References containing label value `9` (T1): 75/76 train/validation, 17/17 test, 92/93 total.
- The only T1-absent reference is `sub-mgh01_301`.
- “T1-positive” means that the reference contains label value `9`; it is not a diagnosis.
- The final handoff references were used regardless of BIDS `desc` entities.
- Legacy `_desc-rater*` and `_desc-staple` files were excluded because they are C2–C8 dorsal inter-rater labels, not the final C2–T1 dorsal-and-ventral target.
- Provenance: [issue #109](https://github.com/ivadomed/model-spinal-rootlets/issues/109), [data-management #392](https://github.com/neuropoly/data-management/issues/392).

## Pairing and crop checks

- Image/label pairs found: 93/93.
- Exact grids: 79/93.
- Lossless axis permutation/header harmonization: 14/93.
- Label interpolation or resampling: none.
- Orientation: RPI under the Spinal Cord Toolbox convention; NiBabel reports `LAS`.
- Detector: `sc-crop 0.6.0`.
- Crop detector input: image only.
- Label use: post-crop foreground-retention and edge-clearance checks only.
- Per-case label-derived crop expansion: none.
- Fixed physical padding: superior 50, inferior 110, left/right 25, anterior 40, posterior 32 mm.
- Crops retaining all label foreground: 93/93.
- Distinct crop shapes: 87.
- Crop/original volume ratio: minimum `0.257`, median `0.365`, maximum `0.607`.
- Minimum label clearance: right `0.6`, left `5.4`, posterior `28.0`, anterior `9.1`, inferior `7.28`, superior `37.6` mm.
- The `0.6 mm` right clearance is limited by the acquisition field of view; the crop cannot extend beyond it.
- Dataset402 moved one crop edge by one voxel after reading a label. Dataset403 does not use that repair: one fixed image-only policy is applied to every case, and any lost foreground fails QC.
- Machine records: [`dataset_manifest.json`](results/dataset403/dataset_manifest.json), [`crop_qc.csv`](results/dataset403/crop_qc.csv).

## Split and training

- The original table placed `sub-005_ses-headUp_007` in validation twice and never validated `sub-010_ses-headNormal_014`.
- The corrected table validates `sub-010_ses-headNormal_014` in fold 1 and keeps `sub-005_ses-headUp_007` in fold 4.
- Every non-test case is now in validation exactly once: 76/76.
- Planner-selected patch: `256 × 128 × 64`.
- Planner-selected spacing: `0.70652 × 0.70000 × 0.70652 mm`.
- Batch size: 2.
- Trainer: `nnUNetTrainer_2000epochsEarlyStopping`.
- Plateau rule: at least 1000 epochs; stop after 400 epochs without an EMA foreground pseudo-Dice gain of at least `0.002`; maximum 2000 epochs.
- Best checkpoints still use nnU-Net’s original unthresholded EMA comparison.

| Run | Validation | Epochs completed | Last material gain | Best EMA | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| fold 0 | 16 | 1277 | 876 | 0.6648 | plateau |
| fold 1 | 15 | 1250 | 849 | 0.6083 | plateau |
| fold 2 | 15 | 1063 | 662 | 0.6578 | plateau |
| fold 3 | 15 | 1333 | 932 | 0.6574 | plateau |
| fold 4 | 15 | 1000 | 495 | 0.6284 | plateau |
| fold_all | all 76 | 2000 | n/a | 0.8194 | maximum |

- `fold_all` pseudo-validation uses its training cases and is not a generalization estimate.
- No traceback, CUDA out-of-memory error, or runtime exception appears in the six logs.
- Native and per-class curves: [`training_curves/`](results/dataset403/training_curves).
- Overview: [`training_curves_overview.jpg`](results/dataset403/figures/training_curves_overview.jpg).
- Checkpoint hashes: [`checkpoint_sha256.txt`](results/dataset403/checkpoint_sha256.txt).
- Plateau definition and replay: [`EARLY_STOPPING.md`](EARLY_STOPPING.md).

## Metric

- Checkpoint: `checkpoint_best.pth` for cropped models.
- Per-level Dice: one full-volume 3-D Dice for each C2–T1 label.
- A missed reference or pure false-positive level scores zero.
- A level absent from both prediction and reference is excluded.
- Per-image macro Dice: unweighted mean of applicable per-level Dice values.
- Binary Dice: all rootlet labels collapsed to foreground.
- Summary: mean ± sample standard deviation.
- Post-processing: none.

## Five-fold out-of-fold results

| Fold | Cases | Macro level Dice | Binary Dice |
| --- | ---: | ---: | ---: |
| 0 | 16 | 0.641 ± 0.096 | 0.660 ± 0.094 |
| 1 | 15 | 0.568 ± 0.101 | 0.598 ± 0.091 |
| 2 | 15 | 0.653 ± 0.062 | 0.661 ± 0.065 |
| 3 | 15 | 0.618 ± 0.059 | 0.641 ± 0.062 |
| 4 | 15 | 0.609 ± 0.090 | 0.621 ± 0.088 |
| All | 76 | 0.618 ± 0.087 | 0.637 ± 0.083 |

| Contrast | Cases | Macro level Dice | Binary Dice |
| --- | ---: | ---: | ---: |
| T2w | 31 | 0.640 ± 0.107 | 0.669 ± 0.097 |
| INV1 | 15 | 0.583 ± 0.056 | 0.593 ± 0.057 |
| INV2 | 15 | 0.627 ± 0.065 | 0.634 ± 0.067 |
| UNIT1 | 15 | 0.599 ± 0.073 | 0.616 ± 0.063 |

| Level | Reference present | Predicted | Dice |
| --- | ---: | ---: | ---: |
| C2 | 76 | 76 | 0.611 ± 0.101 |
| C3 | 76 | 76 | 0.673 ± 0.089 |
| C4 | 76 | 76 | 0.586 ± 0.121 |
| C5 | 76 | 76 | 0.646 ± 0.091 |
| C6 | 76 | 76 | 0.644 ± 0.130 |
| C7 | 76 | 76 | 0.628 ± 0.162 |
| C8 | 76 | 76 | 0.605 ± 0.154 |
| T1 | 75 | 76 | 0.551 ± 0.143 |

- Missing or duplicate out-of-fold predictions: none.
- Machine results: [`cross_validation/`](results/dataset403/cross_validation).

## Matched held-out comparison

- Cohort: the same 17 unchanged source images and final clean references.
- Cropped arm: `fold_all`, `checkpoint_best.pth`.
- Released arm: [`r20250318`](https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20250318), `fold_all/checkpoint_final.pth`.
- Both arms: RPI input, TTA disabled, no post-processing, same Romane RTX A6000 environment.
- Cropped predictions were restored to the source grid before scoring.

| Contrast | n | Released macro | Cropped macro | Released binary | Cropped binary |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 17 | 0.635 ± 0.061 | 0.614 ± 0.064 | 0.644 ± 0.059 | 0.625 ± 0.062 |
| T2w | 5 | 0.639 ± 0.046 | 0.618 ± 0.047 | 0.652 ± 0.035 | 0.638 ± 0.042 |
| INV1 | 4 | 0.603 ± 0.073 | 0.587 ± 0.085 | 0.611 ± 0.072 | 0.595 ± 0.083 |
| INV2 | 4 | 0.661 ± 0.056 | 0.636 ± 0.057 | 0.666 ± 0.057 | 0.642 ± 0.055 |
| UNIT1 | 4 | 0.638 ± 0.079 | 0.616 ± 0.082 | 0.647 ± 0.079 | 0.624 ± 0.081 |

| Level | Released | Cropped | Paired difference |
| --- | ---: | ---: | ---: |
| C2 | 0.614 ± 0.103 | 0.597 ± 0.128 | −0.017 ± 0.043 |
| C3 | 0.608 ± 0.147 | 0.598 ± 0.171 | −0.010 ± 0.047 |
| C4 | 0.609 ± 0.075 | 0.556 ± 0.081 | −0.053 ± 0.038 |
| C5 | 0.637 ± 0.086 | 0.608 ± 0.090 | −0.028 ± 0.058 |
| C6 | 0.670 ± 0.044 | 0.658 ± 0.055 | −0.012 ± 0.047 |
| C7 | 0.667 ± 0.057 | 0.664 ± 0.053 | −0.003 ± 0.021 |
| C8 | 0.665 ± 0.121 | 0.642 ± 0.113 | −0.023 ± 0.037 |
| T1 | 0.612 ± 0.078 | 0.589 ± 0.077 | −0.023 ± 0.034 |

- Both models predicted T1 in all 17 T1-containing references.
- Metrics and paired rows: [`held_out/`](results/dataset403/held_out).
- Quantitative figures: [`test_dice_by_contrast.png`](results/dataset403/figures/test_dice_by_contrast.png), [`test_dice_by_spinal_level.png`](results/dataset403/figures/test_dice_by_spinal_level.png).
- Best/median/worst panels: [`qualitative_best_median_worst.png`](results/dataset403/figures/qualitative_best_median_worst.png).
- All 17 T1-reference panels: [`qualitative_t1_reference_cases.png`](results/dataset403/figures/qualitative_t1_reference_cases.png).

## Complete-cohort inference

- Predictions checked: 93/93.
- Train/validation cases: 76 automatic `fold_all/checkpoint_final` validation outputs.
- Test cases: 17 `fold_all/checkpoint_best` outputs.
- Empty predictions: 0.
- Missing predicted levels: 0.
- C2–C8 were present in all 93 references and predictions.
- T1 was present in 92/93 references and 93/93 predictions; the sole false-positive T1 inventory is the known T1-absent `sub-mgh01_301`.
- The 76-case `fold_all` training score is descriptive and is not used as a generalization result.
- Per-case inventory and descriptive metrics: [`full_cohort/`](results/dataset403/full_cohort), [`held_out/cropped/`](results/dataset403/held_out/cropped).

## Runtime

| Arm | Prepare / detect + crop | nnU-Net | Restore + export | End to end | Per case | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released | 33.40 s | 1101.75 s | 12.08 s | 1147.23 s | 67.48 s | 1.00× |
| Cropped | 480.81 s | 513.66 s | 43.79 s | 1038.25 s | 61.07 s | 1.10× |

- One matched 17-case batch, same Romane RTX A6000 environment, TTA disabled.
- Fresh detector crops were identical in data and grid to the Dataset403 inputs.
- Cropping made nnU-Net itself `2.15×` faster; repeated detector loading reduced the end-to-end gain to `1.10×`.
- Raw records: [`runtime/`](results/dataset403/runtime), [`released_time.txt`](results/dataset403/held_out/released_time.txt), [`cropped_time.txt`](results/dataset403/held_out/cropped_time.txt).

## Reproducibility

- Host/GPU: Romane, NVIDIA RTX A6000, driver 535.288.01.
- Python environment: nnU-Net 2.4.2, PyTorch 2.5.1+cu121, NiBabel 5.4.2, NumPy 2.2.6.
- Tests: `15 passed`.
- Code: dataset preparation, split audit, plateau trainer, plotting, metric calculation, and five-fold consolidation are in this directory.
