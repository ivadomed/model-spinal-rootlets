# Dorsal/ventral rootlet separation

Experimental deterministic post-processing for a RootletSeg output whose label
values encode spinal levels but combine dorsal and ventral rootlets.

## Contract

Inputs:

- a level-labelled rootlet mask;
- a spinal cord mask on the identical voxel grid.

Outputs:

- separate dorsal and ventral masks that retain the input level values;
- optional uncalibrated heuristic score and component-level QC.

The two masks never overlap and their sum is exactly the input label image. The
method therefore partitions existing support; it cannot repair missing or false
positive RootletSeg voxels.

## Method

1. Reorient both masks losslessly to canonical orientation.
2. Estimate a smoothed cord centreline and project the world RAS anterior axis
   onto its local normal plane.
3. Locate each rootlet voxel's nearest cord-surface point.
4. For each component, isolate a narrow adaptive attachment band and divide it
   into connected attachment islands.
5. Assign one dorsal or ventral class to each island from its median local AP
   coordinate, processed separately by level and side.
6. When both seed classes occur in one connected component, propagate identity
   with physical-distance-weighted 3D geodesics inside that fixed component.
7. Retain but flag disconnected components that have no classified attachment.

Posterior/anterior position far from the cord is not treated as ground truth;
pathology can displace distal ventral roots posteriorly.

Known limitations:

- The optional `dense_voxel` v1 strategy can split a thick or oblique branch
  that spans both AP margins. Attachment-island v2 is the default.
- V2 can assign a merged attachment island wholesale or fall back when its
  median AP coordinate is neutral.
- The centerline-normal world RAS frame does not model true cord torsion.
- The heuristic score is a distance margin, not a calibrated probability.
- This is research code and is not ready for anatomical or clinical claims.

## Usage

```bash
python -m postprocessing.dorsal_ventral.split_rootlets \
  --rootlets sub-001_label-rootlets_dseg.nii.gz \
  --cord sub-001_label-SC_seg.nii.gz \
  --output-dorsal sub-001_desc-dorsal_label-rootlets_dseg.nii.gz \
  --output-ventral sub-001_desc-ventral_label-rootlets_dseg.nii.gz \
  --output-score sub-001_desc-dvscore.nii.gz \
  --qc-json sub-001_desc-dvsplit_qc.json
```

## Honest validation plan

There is currently no complete dorsal/ventral reference dataset, so support and
stability checks must not be reported as anatomical accuracy.

Immediate engineering checks:

- exact support, level, grid, and affine preservation;
- zero dorsal/ventral overlap and deterministic reruns;
- stability to orientation and small cord-mask perturbations;
- proximal-seed coverage, fallback rate, and heuristic-score distribution;
- synthetic bridges with a known expected partition.

Partial anatomical evidence:

- Match legacy multi-rater dorsal-only cases to the new combined-label cases.
- On their common support, report dorsal positive agreement and surface
  distance against a consensus, alongside the inter-rater ceiling.
- Treat these reused cases as a debugging reference, not an independent test
  set or publication-level validation cohort.
- Do not call unlabeled or non-overlapping legacy voxels "ventral" and do not
  report ventral Dice from this partial reference.

Minimal true reference set:

- Have an expert label proximal attachments or branches by side and level on a
  small stratified set; full voxel painting is not required initially.
- Report branch-level balanced accuracy, macro-F1 by level/site/contrast,
  score-versus-error coverage curves, and failure categories.
- Fully paint a few hard cases only if voxel-level Dice is needed.

A learned classifier or GNN is justified only after this reference set exposes
repeatable deterministic failure modes. Deterministic outputs may initialize or
regularize a model, but must not be its only targets or its validation truth.

Known-dorsal recall is one-sided. An output that labels every voxel dorsal gets
100% recall while performing no separation, so include that negative control and
never use this audit to rank complete separators.

Run the partial dorsal-positive audit with:

```bash
python -m postprocessing.dorsal_ventral.evaluate_partial_dorsal \
  --combined sub-001_label-rootlets_dseg.nii.gz \
  --predicted-dorsal sub-001_desc-dorsal_label-rootlets_dseg.nii.gz \
  --predicted-ventral sub-001_desc-ventral_label-rootlets_dseg.nii.gz \
  --dorsal-reference sub-001_desc-staple_label-rootlets_dseg.nii.gz \
  --output-json sub-001_desc-partial-dorsal_metrics.json
```

Diagnose whether known dorsal positives are already contaminated at seed time:

```bash
python -m postprocessing.dorsal_ventral.audit_attachment_seeds \
  --combined sub-001_label-rootlets_dseg.nii.gz \
  --cord sub-001_label-SC_seg.nii.gz \
  --dorsal-reference sub-001_desc-staple_label-rootlets_dseg.nii.gz \
  --output-json sub-001_desc-attachment-seed_audit.json
```

Measure assignment sensitivity to one-voxel cord shifts and one-step
erosion/dilation:

```bash
python -m postprocessing.dorsal_ventral.evaluate_cord_perturbations \
  --rootlets sub-001_label-rootlets_dseg.nii.gz \
  --cord sub-001_label-SC_seg.nii.gz \
  --output-json sub-001_desc-cord-perturbation_metrics.json
```

This audit reports changed D/V assignments on the same fixed RootletSeg support.
It measures engineering sensitivity, not correctness.

Export numbered attachment islands and a reviewer CSV without painting full
masks:

```bash
python -m postprocessing.dorsal_ventral.export_attachment_islands \
  --combined sub-001_label-rootlets_dseg.nii.gz \
  --cord sub-001_label-SC_seg.nii.gz \
  --subject sub-001 \
  --output-map sub-001_desc-attachment-islands_dseg.nii.gz \
  --output-csv sub-001_desc-attachment-review.csv
```

After an expert fills `expert_class` with `dorsal` or `ventral`, calculate the
first genuinely supervised metrics with:

```bash
python -m postprocessing.dorsal_ventral.evaluate_attachment_review \
  --review-csv sub-001_desc-attachment-review.csv \
  --review-csv sub-002_desc-attachment-review.csv \
  --output-json attachment-review_metrics.json
```

This reports branch-attachment balanced accuracy and macro-F1 overall and by
subject, level, and side. `unclear` predictions are explicit abstentions: they
reduce coverage and count as errors in headline metrics, while selective
accuracy is reported separately. Empty expert labels are excluded and their
coverage is disclosed. Expert `unclear` decisions count as reviewed but are
excluded from dorsal/ventral scoring, with both review and scorable coverage
reported.

## Paired-session robustness on Romane

`run_marseille_romane.sh` is resumable and uses only the 20 scans already
materialized on Romane. A dry run performs no inference and writes nothing:

```bash
postprocessing/dorsal_ventral/run_marseille_romane.sh --dry-run
```

Resource-intensive execution is intentionally guarded. After booking the
matching Romane GPU and slot:

```bash
RESOURCE_SLOT_BOOKED=1 \
  postprocessing/dorsal_ventral/run_marseille_romane.sh \
  --run --compute gpu --slot 0 --cuda-device 0
```

The GPU preflight refuses to run if SCT's own Python environment cannot see
CUDA; this prevents `SCT_USE_GPU` from silently falling back to CPU. When the
installed SCT environment is CPU-only, reserve CPU capacity and use:

```bash
RESOURCE_SLOT_BOOKED=1 \
  postprocessing/dorsal_ventral/run_marseille_romane.sh \
  --run --compute cpu --slot 0
```

The runner enters the selected resource slice through `set_slot`, skips
complete outputs, validates the exact D/V partition, and writes a manifest plus
paired-session summaries. In GPU mode the slot and CUDA device must match.
It also extracts SCT-reported spinal-cord and RootletSeg runtimes into
`inference_runtime.csv` and `inference_runtime_summary.json`; these exclude the
deterministic splitting stage from SCT's own timers and state the actual
CPU/GPU mode. A separate approximate interval from the RootletSeg output write
to the split-QC write captures SCT return, Python launch, splitting, and output
writes without pretending to be a pure algorithm timer.

The session evaluator does not calculate Dice between unregistered acquisitions.
It reports per-level changes in dorsal fraction, predicted-support volume, level
presence, and fallback fraction. These are engineering stability measures, not
anatomical accuracy. Expert dorsal/ventral attachment labels remain necessary
for balanced accuracy or macro-F1.

The completed runner also creates paired axial PNG mosaics and animated GIFs
under `session_qc/`. These are qualitative failure-screening images;
corresponding slices across sessions are normalized along S/I but are not
registered.
