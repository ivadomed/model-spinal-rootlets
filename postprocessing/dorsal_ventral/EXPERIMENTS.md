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

## 2026-08-02 — visual QC and review pack

- Axial overlays show coherent posterior dorsal / anterior ventral ordering on
  most levels in all three cases.
- Remaining partial-reference disagreements cluster at `sub-amu02` C3–C4,
  `sub-barcelona01` C3–C6, and `sub-brnoUhb03` C3–C5.
- Do not tune those disagreements away without expert attachment labels; distal
  legacy dorsal trajectories can cross the simple AP ordering.
- Numbered attachment review exports contain 56, 55, and 72 islands. Predicted
  class counts are 17/28/11, 20/27/8, and 26/37/9 for
  dorsal/ventral/unclear; only two Brno islands exceed the splitter's maximum
  cord distance.
- Reviewer CSV fields are blank by design. The NIfTI IDs and CSV rows were
  cross-checked one-to-one after export.

## 2026-08-02 — Marseille paired-session robustness run

- Scope stayed fixed to the 20 T2w scans already present on Romane: 10 subjects
  × two sessions. No additional dataset was downloaded or requested.
- All 20 cord, RootletSeg, and D/V outputs completed with no logged errors. Every
  output passed the exact fixed-support partition check; all 10 session pairs and
  all eight C2–T1 levels were present in both sessions.
- SCT's bundled environment reported `torch 2.2.2+cpu`, so the requested CUDA
  path silently fell back to CPU. The resource booking was corrected to CPU and
  GPU 0 released. The runner now requires explicit `--compute cpu|gpu` and a
  CUDA preflight that refuses this fallback in GPU mode.

| Unregistered per-level stability metric, n=80 | Median | IQR | P95 |
| --- | ---: | ---: | ---: |
| Absolute dorsal-fraction difference | 9.1 pp | 35.1 pp | 62.3 pp |
| Rootlet-support relative-volume difference | 11.9% | 13.3% | 36.0% |
| Absolute fallback-fraction difference | 4.2 pp | 25.0 pp | 33.3 pp |

- The largest dorsal-fraction changes were `sub-08` C3 (83.2 pp), `sub-05` C2
  (68.2 pp), and `sub-07` C4 (67.4 pp). For `sub-08` C3, support volume changed
  by only 0.8%, so volume stability alone would miss the class switch.
- Across all 20 scans, 94/669 components (14.1%) used a fallback. Predicted
  dorsal support ranged from 18.0% to 82.9% across scans (median 61.0%).
- Paired axial PNGs and GIFs were generated for all 10 subjects. They are
  qualitative failure screens at normalized S/I positions, not registered
  comparisons or anatomical truth.

| CPU timing, n=20 | Median | IQR | P95 |
| --- | ---: | ---: | ---: |
| SCT spinal-cord command | 27.35 s | 0.19 s | 27.87 s |
| SCT RootletSeg command | 88.65 s | 0.24 s | 88.86 s |
| Combined SCT commands | 115.94 s | 0.35 s | 116.44 s |
| Rootlet output → split-QC write interval | 2.22 s | 0.09 s | 2.33 s |

- The last interval is approximate: it includes SCT return, Python launch, the
  deterministic split, and output writes; it is not a pure algorithm timer.
- Decision: v2 is useful as a deterministic baseline, review prefill, and source
  of hard-case candidates, but session brittleness rules out treating its labels
  as ground truth. Prioritize expert attachment review at C3–C5, including
  `sub-08` C3, before fitting a small classifier or GNN.

## 2026-08-02 — cord-mask perturbation audit

- Perturbations: one canonical-RAS voxel in each right/anterior direction plus
  one binary erosion and dilation. Rootlet support remains fixed.
- Metric: fraction of RootletSeg voxels whose deterministic D/V assignment
  changes versus the unperturbed cord mask. This is stability, not accuracy.

| Case | Mean flip rate | Worst flip rate | Worst perturbation |
| --- | ---: | ---: | --- |
| `sub-amu02` | 9.7% | 13.3% | anterior −1 voxel |
| `sub-barcelona01` | 4.2% | 17.3% | anterior +1 voxel |
| `sub-brnoUhb03` | 3.6% | 7.3% | right −1 voxel |

- Highest per-level flip rates occur at C3–C5: 46.0% at AMU C3, 46.1% at
  Barcelona C3, and 29.8% at Brno C3.
- Interpretation: aggregate stability is moderate, but individual attachment
  decisions can switch wholesale under a one-voxel cord change. These levels
  overlap earlier visual disagreement zones and should be prioritized for
  expert attachment review.

## 2026-08-02 — side-paired attachment v3 candidate

- Failure diagnosis: v2 uses a fixed ±0.5 mm AP margin independently for every
  component. Neutral islands and whole-component fallbacks switched between
  sessions. A first pooled v3 prototype removed that absolute origin, but a
  tiny extreme island at `sub-07` C6 could move the boundary for both sides.
- Change: fit a deterministic weighted two-mode AP boundary within each side
  and level. The posterior mode is dorsal. If one side has fewer than two
  candidates or less than 0.5 mm span, use the pooled level boundary; if that
  is also unavailable, retain the v2 absolute-margin fallback.
- Scope: the same 20 Marseille masks and cord segmentations, plus the same
  three legacy known-dorsal cases. Four targeted Marseille T2w files (18 MB,
  `sub-13`/`sub-14`, both sessions) were copied only for worst-case visual QC.
  No new cohort was cloned or annexed.

| Marseille per-level session metric, n=80 | V2 | Pooled v3 | Side-paired v3 |
| --- | ---: | ---: | ---: |
| Median absolute dorsal-fraction difference | 9.1 pp | 4.5 pp | 4.5 pp |
| IQR | 35.1 pp | 8.8 pp | 8.4 pp |
| P95 | 62.3 pp | 29.7 pp | 26.8 pp |

| Marseille cord perturbation summary, n=20 | V2 | Pooled v3 | Side-paired v3 |
| --- | ---: | ---: | ---: |
| Median per-scan mean flip rate | 7.5% | 0.8% | 0.5% |
| P95 per-scan mean flip rate | 15.1% | 5.9% | 4.6% |
| Median per-scan worst flip rate | 21.4% | 2.9% | 2.4% |
| Cohort worst flip rate | 35.9% | 18.8% | 16.1% |

- Side-paired v3 used a side-specific boundary for 301/320 side-level
  decisions and a pooled fallback for 19/320; no side-level decision required
  the absolute boundary. Twelve of 669 components still lacked a classified
  attachment and used the component fallback.
- The remaining largest session changes are `sub-14` C3 (39.4 pp), `sub-13`
  C3 (33.1 pp), and `sub-06` C3 (31.1 pp). QC shows missing or changed
  attachment topology across sessions, not a single tunable global threshold.
- Targeted overlays retain posterior/anterior ordering, but they are
  unregistered qualitative screens and cannot determine which session is
  anatomically correct.

| Reused known-dorsal case | Recall | Dorsal support | Enrichment over all-dorsal baseline |
| --- | ---: | ---: | ---: |
| `sub-amu02` | 99.4% | 61.2% | 1.62× |
| `sub-barcelona01` | 100.0% | 71.0% | 1.41× |
| `sub-brnoUhb03` | 100.0% | 62.1% | 1.61× |

- This table is a one-sided leakage sanity check, not accuracy. The all-dorsal
  control has 100% recall. Marseille v3 assigns a median 78.9% of RootletSeg
  voxels dorsal (range 70.8–86.8%), so apparent stability must not be used to
  dismiss possible class imbalance or collapse.
- Decision: retain v2 as the compatibility baseline and expose side-paired v3
  as the engineering candidate for Jan's attachment review. Do not train a GNN
  or declare v3 superior anatomically until both dorsal and ventral expert
  labels produce balanced accuracy/macro-F1 and coverage results.
