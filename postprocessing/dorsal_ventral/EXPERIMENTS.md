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
