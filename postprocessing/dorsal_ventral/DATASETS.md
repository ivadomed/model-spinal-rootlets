# Dataset inventory

## Anatomical debugging

- Three matched cases: `sub-amu02`, `sub-barcelona01`, `sub-brnoUhb03`.
- Available: combined C2–T1 mask, cord mask, four legacy dorsal-only raters,
  STAPLE dorsal reference.
- Use: known-dorsal leakage and branch-error inspection.
- Limitation: reused dorsal-positive data, no ventral ground truth, not an
  independent test set.

## Robustness cohorts

- `hc-leipzig-7t-mp2rage`: 7T MP2RAGE cohort already materialized on Romane;
  rootlet labels are present. Use for contrast consistency and oblique-grid
  stress tests after generating or locating matching cord masks.
- `marseille-rootlets`: restricted `data.neuro` dataset
  ([repository](https://data.neuro.polymtl.ca/datasets/marseille-rootlets.git))
  cloned and fully annexed at
  `/home/kuanyiw/projects/rootlets/data/marseille-rootlets` on Romane.
  It contains 10 healthy controls, two sessions each, at 0.8 mm isotropic T2w.
  Use for session consistency and splitter failure/fallback rates.
- `ds004507`: public head-position cohort already cloned on Romane. Use paired
  head-up/head-normal/head-down scans to measure posture sensitivity once the
  required annex payloads and segmentations are selected.

None of these robustness cohorts provides complete dorsal/ventral truth. Their
consistency metrics complement, but cannot replace, a small expert-labelled
branch/attachment reference set.

Current acquisition boundary: do not fetch or annex another cohort for this
prototype. Marseille is sufficient to exercise the paired-session pipeline;
additional unlabeled images would add compute volume without resolving the
anatomical validation gap.
