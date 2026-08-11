# [Results] Dorsal/ventral rootlet separation (hybrid V5)

## Summary

- V5 separates clear rootlet clusters by their mean anterior/posterior position in RPI.
- Ambiguous or merged levels are sent to a 3-D D/V network; all outputs remain inside the original RootletSeg mask.
- On three held-out expert-labelled cases, the 3-D classifier alone scored **6,071/6,112 voxels (99.33%)**; the requested hybrid scored **6,068/6,112 (99.28%)**.
- The best fully deterministic baseline was V3 at **5,961/6,112 voxels (97.53%)** in **0.25 s/scan**, excluding its shared cord-mask prerequisite.
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
- V2-V4 use the same automatic cord mask: SCT `deepseg_sc`, T1, SVM centreline, 2-D kernel, after a 20 mm rootlet-bounded crop; no mask was manually corrected.
- The V5 maps were regenerated from the frozen checkpoint for this review and reproduced every locked test metric exactly.
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
| Unit tests | 48/48 |
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
| CPU core time | 3.85 s/scan |

V5 alone is precise when it accepts a level, but it deliberately leaves 28.3% of held-out rootlet voxels for the fallback.

### Learned fallback

| Validation candidate | Routed-region accuracy | Merged-region accuracy | Balanced accuracy | Mean D/V Dice |
| --- | ---: | ---: | ---: | ---: |
| 2-D nnU-Net | 98.44% | N/A (no merged validation component) | 98.88% | 98.91% |
| 3-D nnU-Net | 99.83% | N/A (no merged validation component) | 99.95% | 99.88% |

- Selected: **3-D full-resolution nnU-Net**.
- Rule: routed-region accuracy → merged-region accuracy → global balanced accuracy → mean D/V Dice.

### Frozen V2-V5 comparison

| Method | Voxel accuracy | Balanced accuracy | Dorsal Dice | Ventral Dice | Simple components | Merged-region voxels | D/V time/scan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 attachment islands | 84.90% | 87.46% | 88.80% | 76.84% | 91/101 | 73.86% | 0.21 s |
| V3 side-paired | 97.53% | 96.10% | 98.32% | 95.30% | 99/101 | 73.86% | 0.25 s |
| V4 attachment graph | 97.02% | 95.76% | 97.97% | 94.39% | 99/101 | 73.86% | 0.25 s |
| V5 hybrid | 99.28% | 98.66% | 99.51% | 98.65% | 100/101 | 91.36% | 18.92 s |
| 3-D classifier alone | **99.33%** | **98.75%** | **99.54%** | **98.74%** | **100/101** | **91.36%** | 14.56 s |

- Same three untouched cases, 6,112 manual rootlet voxels, and one expert target.
- V2-V4 share one automatic SCT cord mask. Their 0.21-0.25 s timings exclude cord preparation; cropped cord preparation took 9.75-10.32 s/scan.
- V3 is the strongest no-training method. V4 added complexity without improving this test.
- The classifier alone remained marginally better than the hybrid by 3/6,112 voxels.
- The test is too small for a general accuracy claim.

### External inference

| Dataset | Scans | Mean V5 level coverage | Mean voxels sent to fallback | Exact support |
| --- | ---: | ---: | ---: | ---: |
| ds004507 | 2 | 62.5% | 34.2% | 2/2 |
| spine-generic | 3 | 54.2% | 48.7% | 3/3 |
| HC-Leipzig | 12 | 78.1% | 16.4% | 12/12 |
| Marseille | 20 | 45.0% | 56.0% | 20/20 |

No external D/V ground truth exists, so external Dice or accuracy is not reported.

### Inference speed

| Mode | Wall time per scan |
| --- | ---: |
| Warm A6000 batch, cropped scans, probabilities saved | 2.96 s |
| Warm A6000 batch, mixed crop sizes | 7.21–7.85 s |
| Cold A6000 process, cropped RootletSeg mask | 16.58–16.88 s |
| Cold A6000 process, larger manual mask grid | 23.97–24.18 s |
| Tassan RTX Pro 6000 recovery run, same 3 manual-mask cases | 37.28 s |
| Cold CPU process, cropped RootletSeg mask | 180.06 s |
| Deterministic V5 CPU command | 1.86–5.52 s |
| V5/fallback combination CPU command | 2.15–6.84 s |

- Practical budget: **3–10 s/scan** with a warm GPU worker, or **15–30 s** for a one-off GPU command.
- The 37.28 s Tassan recovery used a fresh compatibility environment to reproduce the locked maps; it is reported for provenance, not as the deployment baseline.
- The label source does not change the model; crop/grid size and startup explain the manual-versus-RootletSeg difference.
- Human annotation and upstream RootletSeg inference time are excluded.

## Method

- **V1:** one global anterior/posterior cut; fast, but ignores sides and individual rootlets.
- **V2:** find cord-attachment seeds and spread their labels through each rootlet; can split one branch incorrectly.
- **V3:** pair attachment islands on each side before spreading labels; better topology, but depends on a reliable cord mask.
- **V4:** represent attachments as a graph; flexible, but still inherits uncertain attachment seeds.
- **V5:** count rootlet clusters first. Clear clusters are sorted by mean RPI `y`; only unclear levels use the learned fallback.

The learned model does not determine rootlet support from scratch. It produces a dense background/dorsal/ventral map from the MRI and existing level-labelled rootlet mask; the final output keeps only dorsal or ventral decisions inside that fixed mask.

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
- Repeated inference timings: `results/v5/metrics/inference_speed.json`
- V2-V5 metrics: `results/v5/metrics/v2-v5-comparison.{json,csv}`
- V2-V5 accuracy, speed, cord QC, montages, and GIF: `results/v5/artifacts/v2-v5/`
- Checkpoint manifest: `results/v5/model/model_manifest.json`
- Reproduction commands: `README.md`
