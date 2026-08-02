#!/usr/bin/env python3
"""Evaluate a split against a dorsal-only positive reference.

This intentionally reports no ventral accuracy. Reference-negative voxels are
unknown because the historical annotation contains dorsal rootlets only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_partial_dorsal(
    combined: np.ndarray,
    predicted_dorsal: np.ndarray,
    predicted_ventral: np.ndarray,
    dorsal_reference: np.ndarray,
) -> dict[str, Any]:
    named_arrays = {
        "combined": combined,
        "predicted_dorsal": predicted_dorsal,
        "predicted_ventral": predicted_ventral,
        "dorsal_reference": dorsal_reference,
    }
    if any(array.shape != combined.shape for array in named_arrays.values()):
        raise ValueError("All arrays must have the same shape.")
    arrays = []
    for name, array in named_arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        rounded = np.rint(array)
        if not np.allclose(array, rounded, atol=1e-4):
            raise ValueError(f"{name} must be integer-valued; check label resampling.")
        if np.any(rounded < 0):
            raise ValueError(f"{name} must be non-negative.")
        arrays.append(rounded.astype(np.int32))
    combined, predicted_dorsal, predicted_ventral, dorsal_reference = arrays
    if np.any((predicted_dorsal > 0) & (predicted_ventral > 0)):
        raise ValueError("Predicted dorsal and ventral masks overlap.")
    if not np.array_equal(predicted_dorsal + predicted_ventral, combined):
        raise ValueError("Predicted dorsal and ventral masks do not partition the combined mask.")

    def metrics_for_mask(label_mask: np.ndarray) -> dict[str, Any]:
        reference_positive = (dorsal_reference > 0) & label_mask
        combined_positive = (combined > 0) & label_mask
        common_positive = reference_positive & combined_positive
        reference_voxels = int(np.count_nonzero(reference_positive))
        common_voxels = int(np.count_nonzero(common_positive))
        dorsal_hits = int(np.count_nonzero((predicted_dorsal > 0) & common_positive))
        ventral_leaks = int(np.count_nonzero((predicted_ventral > 0) & common_positive))
        level_matches = int(
            np.count_nonzero((predicted_dorsal == dorsal_reference) & common_positive)
        )
        return {
            "reference_voxels": reference_voxels,
            "reference_voxels_in_combined_support": common_voxels,
            "reference_coverage": _safe_ratio(common_voxels, reference_voxels),
            "known_dorsal_recall": _safe_ratio(dorsal_hits, common_voxels),
            "known_dorsal_leakage_to_ventral": _safe_ratio(ventral_leaks, common_voxels),
            "known_dorsal_level_agreement": _safe_ratio(level_matches, common_voxels),
        }

    result = metrics_for_mask(np.ones(combined.shape, dtype=bool))
    combined_voxels = int(np.count_nonzero(combined))
    result.update(
        {
            "combined_voxels": combined_voxels,
            "predicted_dorsal_voxels": int(np.count_nonzero(predicted_dorsal)),
            "predicted_ventral_voxels": int(np.count_nonzero(predicted_ventral)),
            "predicted_dorsal_fraction": _safe_ratio(
                int(np.count_nonzero(predicted_dorsal)), combined_voxels
            ),
        }
    )
    per_label = []
    for label in np.unique(dorsal_reference[dorsal_reference > 0]):
        label_metrics = metrics_for_mask(dorsal_reference == label)
        label_metrics["label"] = int(label)
        per_label.append(label_metrics)
    result["labels"] = per_label
    result["limitations"] = [
        "Dorsal-only reference: ventral accuracy and specificity are not identifiable.",
        "An all-dorsal output scores perfect known-dorsal recall, so this cannot rank separators.",
        "Reference coverage measures support agreement and must be interpreted separately.",
    ]
    return result


def _load_same_grid(paths: list[str]) -> list[nib.Nifti1Image]:
    images = [nib.load(path) for path in paths]
    reference = images[0]
    for image, path in zip(images[1:], paths[1:]):
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-4
        ):
            raise ValueError(f"Image is not on the combined-mask grid: {path}")
    return images


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--predicted-dorsal", required=True)
    parser.add_argument("--predicted-ventral", required=True)
    parser.add_argument("--dorsal-reference", required=True)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    paths = [
        args.combined,
        args.predicted_dorsal,
        args.predicted_ventral,
        args.dorsal_reference,
    ]
    images = _load_same_grid(paths)
    metrics = evaluate_partial_dorsal(
        *(np.asanyarray(image.dataobj) for image in images)
    )
    metrics["inputs"] = {
        "combined": str(Path(args.combined).resolve()),
        "predicted_dorsal": str(Path(args.predicted_dorsal).resolve()),
        "predicted_ventral": str(Path(args.predicted_ventral).resolve()),
        "dorsal_reference": str(Path(args.dorsal_reference).resolve()),
    }
    rendered = json.dumps(metrics, indent=2) + "\n"
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
