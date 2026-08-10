# [Results] Dorsal/ventral rootlet separation (hybrid V5)

## Summary

- V5 separates clear rootlet clusters by their mean anterior/posterior position in RPI.
- Ambiguous or merged levels are sent to a small D/V network; all outputs remain inside the original RootletSeg mask.
- On three held-out expert-labelled cases, the 3-D classifier alone scored **6,071/6,112 voxels (99.33%)**; the requested hybrid scored **6,068/6,112 (99.28%)**.
- V5 is useful as an interpretable confidence/QC gate, but it did not improve accuracy on this small test set.
- This is a 15-case pilot, not a release model.

## Data

| Use | Images | Rootlet input | D/V target |
| --- | ---: | --- | --- |
| Train | 10 HC-Leipzig UNIT1 | Manual level labels | One expert |
| Validation | 2 HC-Leipzig UNIT1 | Manual level labels | One expert |
| Test | 3 HC-Leipzig UNIT1 | Manual level labels | One expert |
| External QC | 37 | RootletSeg predictions | None |

- The split was fixed by a SHA-256 case ranking before model training.
- The 15 expert cases contain 517 components, including 15 components manually split into both classes.
- Test accuracy is conditional on the manual rootlet mask; it is not end-to-end RootletSeg accuracy.
- External QC uses 2 ds004507, 3 spine-generic, 12 HC-Leipzig, and 20 scans from the restricted [Marseille repository](https://data.neuro.polymtl.ca/datasets/marseille-rootlets.git).
- ds004507, spine-generic, and Marseille are unseen D/V datasets. The HC-Leipzig QC scans are source-domain and include overlap with the 15 labelled cases.

## Checks

| Check | Result |
| --- | --- |
| Orientation | RPI |
| Split | Case-wise 10/2/3 |
| Architecture choice | Frozen validation rule |
| Test access | Once, after architecture selection |
| Output support | Dorsal + ventral = RootletSeg exactly |
| Unit tests | 42/42 |
| External scans | 37 staged; 4 datasets |

## Results

### Deterministic V5 only

| Held-out result | Value |
| --- | ---: |
| Levels handled automatically | 15/21 (71.4%) |
| Components handled automatically | 77/101 (76.2%) |
| Component accuracy where used | 76/77 (98.7%) |
| Voxel accuracy where used | 4,379/4,384 (99.9%) |
| Merged components routed to fallback | 2/2 |

V5 alone is precise when it accepts a level, but it deliberately leaves 28.3% of held-out rootlet voxels for the fallback.

### Learned fallback

| Validation candidate | Routed-region accuracy | Merged-region accuracy | Balanced accuracy | Mean D/V Dice |
| --- | ---: | ---: | ---: | ---: |
| 2-D nnU-Net | 98.44% | N/A (no merged validation component) | 98.88% | 98.91% |
| 3-D nnU-Net | 99.83% | N/A (no merged validation component) | 99.95% | 99.88% |

- Selected: **3-D full-resolution nnU-Net**.
- Rule: routed-region accuracy → merged-region accuracy → global balanced accuracy → mean D/V Dice.

### Final held-out test

| Method | Voxel accuracy | Balanced accuracy | Dorsal Dice | Ventral Dice | Merged-region accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3-D classifier alone | 99.33% | 98.75% | 99.54% | 98.74% | 91.36% |
| Hybrid V5 | 99.28% | 98.66% | 99.51% | 98.65% | 91.36% |

Both methods classified 100/101 simple components correctly. The three-voxel difference favours the classifier alone; it is too small and the test set is too limited to support a general claim.

### External inference

| Dataset | Scans | Mean V5 level coverage | Mean voxels sent to fallback | Exact support |
| --- | ---: | ---: | ---: | ---: |
| ds004507 | 2 | 62.5% | 34.2% | 2/2 |
| spine-generic | 3 | 54.2% | 48.7% | 3/3 |
| HC-Leipzig | 12 | 78.1% | 16.4% | 12/12 |
| Marseille | 20 | 45.0% | 56.0% | 20/20 |

No external D/V ground truth exists, so external Dice or accuracy is not reported.

- 3-D classifier: 109.51 s for 37 scans (2.96 s/scan, batch startup included).
- V5 combination: 227.17 s on CPU (6.14 s/scan).
- Total D/V add-on: 336.68 s (9.10 s/scan); upstream RootletSeg time is excluded.

## Method

- **V1:** one global anterior/posterior cut; fast, but ignores sides and individual rootlets.
- **V2:** find cord-attachment seeds and spread their labels through each rootlet; can split one branch incorrectly.
- **V3:** pair attachment islands on each side before spreading labels; better topology, but depends on a reliable cord mask.
- **V4:** represent attachments as a graph; flexible, but still inherits uncertain attachment seeds.
- **V5:** count rootlet clusters first. Clear clusters are sorted by mean RPI `y`; only unclear levels use the learned fallback.

The learned model does not segment rootlets again. It predicts dorsal or ventral only inside an existing RootletSeg mask, and only its prediction inside V5-routed levels is used.

## Recommendation

- Use the 3-D classifier as the current D/V output: it was marginally better on the held-out cases.
- Keep V5 to expose easy versus ambiguous levels and to trigger QC; do not claim it improves accuracy yet.
- Label more independent cases before retraining or considering a graph network.

## Limits

- 15 labelled cases, one rater, one source dataset, and a three-case test set.
- Manual rootlet masks were used for quantitative D/V evaluation.
- External RootletSeg predictions have no expert D/V labels.
- The current result supports a pilot and more annotation; it does not establish clinical generalization.

## Review

- Held-out metric table: `results/v5/artifacts/heldout/heldout_per_case.csv`
- Held-out metrics: `results/v5/artifacts/heldout/heldout_metrics.png`
- Held-out axial GIF: `results/v5/artifacts/heldout/heldout_median_accuracy_sweep.gif`
- Per-case held-out overlays: `results/v5/artifacts/heldout/heldout_case_*_montage.png`
- External dataset GIFs/PNGs: `results/v5/artifacts/external/`
- Training curves: `results/v5/artifacts/heldout/fallback_training_curves.png`
- Participant-free metrics: `results/v5/metrics/public_summary.json`
- Checkpoint manifest: `results/v5/model/model_manifest.json`
- Reproduction commands: `README.md`
