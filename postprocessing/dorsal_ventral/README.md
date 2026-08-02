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
4. Seed dorsal and ventral identity from posterolateral and anterolateral
   proximal attachments, processed separately by level and side.
5. When both seed classes occur in one connected component, propagate identity
   with physical-distance-weighted 3D geodesics inside that fixed component.
6. Retain but flag disconnected components that have no proximal seed.

Posterior/anterior position far from the cord is not treated as ground truth;
pathology can displace distal ventral roots posteriorly.

Known v1 limitations:

- A thick or oblique single branch may span both AP seed margins and be split as
  if it were a joined dorsal/ventral component.
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
