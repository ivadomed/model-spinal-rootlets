# Hybrid V5 evidence package

- `metrics/public_summary.json`: participant-free validation, held-out test,
  external QC, and runtime aggregates.
- `artifacts/heldout/`: three anonymized expert comparisons, one 24-frame GIF,
  architecture curves, metrics, and the pipeline diagram.
- `artifacts/external/`: one representative montage and 24-frame GIF per
  dataset, plus cohort QC and a CSV summary.
- `model/model_manifest.json`: selected architecture and checkpoint checksum.

All axial displays are RPI with anterior at the top; blue is ventral and red is
dorsal. External scans have no expert D/V labels, so their figures show
qualitative QC and routing coverage—not accuracy. The 236 MB checkpoint is
stored locally outside Git and can be verified with the manifest SHA-256.
