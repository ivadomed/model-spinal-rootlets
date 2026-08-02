# Experiment log

## 2026-08-02 — proximal-attachment geodesic v1

- Goal: partition fixed RootletSeg support without training labels.
- Data: combined C2–T1 masks plus cord masks; three legacy cases have four
  dorsal-only raters and a STAPLE reference on the same rootlet grid.
- The legacy masks are not an independent held-out dataset. They are used only
  as known-dorsal positives for debugging, not as headline validation.
- Parameters: 4.0 mm attachment distance; 0.5 mm AP seed margin.
- Invariants: exact support and level preservation; zero overlap; deterministic
  RAS/non-RAS tests pass.
- Partial metric: recall of known dorsal-positive voxels within the combined
  mask. This is not ventral accuracy.

| Case | STAPLE coverage | Known-dorsal recall | Dorsal support fraction |
| --- | ---: | ---: | ---: |
| `sub-amu02` | 76.8% | 60.1% | 36.0% |
| `sub-barcelona01` | 92.4% | 52.1% | 35.0% |
| `sub-brnoUhb03` | 94.7% | 71.2% | 45.3% |

- `sub-brnoUhb03` required nearest-neighbour resampling of the cord mask from
  0.8 mm in-plane to the 0.5 mm rootlet/reference grid; the rootlet and dorsal
  reference masks themselves already shared a grid.
- Negative control: an all-dorsal partition has 100% known-dorsal recall while
  performing no separation. This metric can expose definite dorsal leakage but
  cannot rank complete separators or identify ventral accuracy.
- Decision: v1 fails the one-sided known-dorsal sanity audit. Diagnose attachment
  seeds and component bridges before considering a learned model.
- Next evidence needed: expert branch/attachment labels that include both dorsal
  and ventral classes. Do not train a GNN solely from deterministic pseudo-labels.

## 2026-08-02 — attachment-seed audit

- Question: does known-dorsal leakage begin at seed construction or later during
  geodesic propagation?
- Scope: STAPLE dorsal-positive voxels that overlap the combined mask and lie
  within 4.0 mm of the cord.

| Case | Proximal known-dorsal coverage | Dorsal seed | Ventral seed | Neutral |
| --- | ---: | ---: | ---: | ---: |
| `sub-amu02` | 68.5% | 40.4% | 26.6% | 33.0% |
| `sub-barcelona01` | 68.0% | 42.4% | 26.0% | 31.6% |
| `sub-brnoUhb03` | 62.7% | 55.1% | 14.1% | 30.9% |

- Components containing both seed classes hold 679/867, 1814/2234, and
  2346/4367 overlapping known-dorsal voxels, respectively.
- Interpretation: dense voxel seeding is contaminated before propagation. A
  thick/oblique branch can span both margins and be bisected incorrectly.
- Parameter sensitivity confirms a trade-off rather than a fix. Across the
  three cases, reducing the attachment band from 4.0 to 2.0 mm and increasing
  the AP margin from 0.5 to 1.0 mm lowers mean known-dorsal ventral seeding from
  22.2% to 2.9%, but proximal coverage falls from 66.4% to 27.7% and 52.7% of
  those proximal positives become neutral.
- Next deterministic version: extract branch attachment nodes first; assign one
  class per attachment/branch; use graph propagation only downstream.

## 2026-08-02 — attachment-island geodesic v2

- Change: use a 0.8 mm adaptive band from each component's closest cord point;
  group it into connected islands; assign one median-AP class per island.
- Same masks, references, and one-sided metric as v1.

| Case | V1 known-dorsal recall | V2 known-dorsal recall | V2 dorsal fraction | V2 fallbacks |
| --- | ---: | ---: | ---: | ---: |
| `sub-amu02` | 60.1% | 82.6% | 51.3% | 6 |
| `sub-barcelona01` | 52.1% | 78.9% | 57.2% | 3 |
| `sub-brnoUhb03` | 71.2% | 81.5% | 51.2% | 4 |

- Dual-class geodesic components fall to one, one, and zero respectively; v2
  avoids many dense-seed bisections and exposes neutral attachments as fallbacks.
- Predicted-dorsal overlap with the partial reference is enriched 1.61×, 1.38×,
  and 1.59× over the corresponding all-dorsal support baseline. This guards
  against interpreting recall alone, but it is still not precision.
- Interpretation: better one-sided sanity behavior, not validated separation.
  The all-dorsal control still has 100% known-dorsal recall, and ventral truth is
  unavailable.
