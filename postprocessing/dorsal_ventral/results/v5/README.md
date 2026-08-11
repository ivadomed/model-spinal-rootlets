# Hybrid V5 evidence package

- `metrics/public_summary.json`: participant-free validation, held-out test,
  external QC, and runtime aggregates.
- `metrics/inference_speed.json`: repeated cold GPU, warm batch, CPU, V5, and
  hybrid timing measurements.
- `metrics/v2-v5-comparison.{json,csv}`: frozen-test V2-V5 accuracy, coverage,
  component checks, runtime, and prerequisites.
- `artifacts/heldout/`: three anonymized expert comparisons, one 24-frame GIF,
  architecture curves, metrics, and the pipeline diagram.
- `artifacts/external/`: one representative montage and 24-frame GIF per
  dataset, plus cohort QC and a CSV summary.
- `artifacts/v2-v5/`: accuracy and speed charts, three anonymized method
  montages, shared cord-input QC, a V5 decision-path diagram, and
  failure-focused grids/GIFs. The latter show slices with a V2-V4 error and
  distinguish a correct V5 gate decision from an amber V5 abstention resolved
  by the learned outputs.
- `model/model_manifest.json`: selected architecture and checkpoint checksum.

All axial displays are RPI with anterior at the top; blue is ventral and red is
dorsal. External scans have no expert D/V labels, so their figures show
qualitative QC and routing coverage—not accuracy. The 236 MB checkpoint is
stored locally outside Git and can be verified with the manifest SHA-256.
