#!/usr/bin/env python3
"""Measure deterministic split sensitivity to small spinal-cord mask changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from postprocessing.dorsal_ventral.split_rootlets import (
    _to_canonical,
    split_rootlets,
)


PERTURBATIONS = ("right_minus_1", "right_plus_1", "anterior_minus_1", "anterior_plus_1", "erode_1", "dilate_1")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def perturb_cord(cord: np.ndarray, name: str) -> np.ndarray:
    mask = cord > 0
    if name == "erode_1":
        result = ndimage.binary_erosion(mask, iterations=1)
    elif name == "dilate_1":
        result = ndimage.binary_dilation(mask, iterations=1)
    else:
        shifts = {
            "right_minus_1": (-1, 0, 0),
            "right_plus_1": (1, 0, 0),
            "anterior_minus_1": (0, -1, 0),
            "anterior_plus_1": (0, 1, 0),
        }
        if name not in shifts:
            raise ValueError(f"Unknown perturbation: {name}")
        result = ndimage.shift(
            mask.astype(np.uint8),
            shift=shifts[name],
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ) > 0
    if not np.any(result):
        raise ValueError(f"Perturbation {name} produced an empty cord mask.")
    return result


def _comparison(
    rootlets: np.ndarray,
    baseline_dorsal: np.ndarray,
    candidate_dorsal: np.ndarray,
) -> dict[str, Any]:
    support = rootlets > 0
    flipped = support & ((baseline_dorsal > 0) != (candidate_dorsal > 0))

    def record(mask: np.ndarray) -> dict[str, Any]:
        support_voxels = int(np.count_nonzero(mask & support))
        flip_voxels = int(np.count_nonzero(mask & flipped))
        baseline_dorsal_voxels = int(np.count_nonzero(mask & (baseline_dorsal > 0)))
        candidate_dorsal_voxels = int(np.count_nonzero(mask & (candidate_dorsal > 0)))
        return {
            "support_voxels": support_voxels,
            "flip_voxels": flip_voxels,
            "flip_fraction": _safe_ratio(flip_voxels, support_voxels),
            "baseline_dorsal_fraction": _safe_ratio(
                baseline_dorsal_voxels, support_voxels
            ),
            "candidate_dorsal_fraction": _safe_ratio(
                candidate_dorsal_voxels, support_voxels
            ),
            "abs_dorsal_fraction_difference": _safe_ratio(
                abs(candidate_dorsal_voxels - baseline_dorsal_voxels), support_voxels
            ),
        }

    metrics = record(support)
    metrics["levels"] = {
        str(int(level)): record(rootlets == level)
        for level in np.unique(rootlets[support])
    }
    return metrics


def evaluate_cord_perturbations(
    rootlets: np.ndarray,
    cord: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    affine: np.ndarray | None = None,
) -> dict[str, Any]:
    baseline = split_rootlets(rootlets, cord, spacing, affine=affine)
    variants: dict[str, Any] = {}
    for name in PERTURBATIONS:
        candidate = split_rootlets(
            rootlets,
            perturb_cord(cord, name),
            spacing,
            affine=affine,
        )
        metrics = _comparison(rootlets, baseline.dorsal, candidate.dorsal)
        metrics["fallback_components"] = candidate.qc["fallback_components"]
        metrics["fallback_component_difference"] = int(
            candidate.qc["fallback_components"] - baseline.qc["fallback_components"]
        )
        variants[name] = metrics
    flip_values = [float(item["flip_fraction"] or 0.0) for item in variants.values()]
    worst_name = max(variants, key=lambda name: float(variants[name]["flip_fraction"] or 0.0))
    return {
        "interpretation": (
            "Engineering sensitivity to small cord-mask perturbations only; not anatomical accuracy."
        ),
        "baseline_fallback_components": baseline.qc["fallback_components"],
        "mean_flip_fraction": float(np.mean(flip_values)),
        "maximum_flip_fraction": float(max(flip_values)),
        "worst_perturbation": worst_name,
        "perturbations": variants,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rootlet_image = nib.load(args.rootlets)
    cord_image = nib.load(args.cord)
    if rootlet_image.shape != cord_image.shape or not np.allclose(
        rootlet_image.affine, cord_image.affine, atol=1e-4
    ):
        raise ValueError("Rootlet and cord images must share a voxel grid and affine.")
    rootlets, rootlet_transform = _to_canonical(
        np.asanyarray(rootlet_image.dataobj), rootlet_image.affine
    )
    cord, cord_transform = _to_canonical(
        np.asanyarray(cord_image.dataobj), cord_image.affine
    )
    if not np.array_equal(rootlet_transform, cord_transform):
        raise ValueError("Rootlet and cord orientation transforms differ.")
    canonical_affine = rootlet_image.affine @ nib.orientations.inv_ornt_aff(
        rootlet_transform, rootlet_image.shape
    )
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(canonical_affine))
    metrics = evaluate_cord_perturbations(
        rootlets, cord, spacing, affine=canonical_affine
    )
    metrics.update(
        {
            "rootlets": str(Path(args.rootlets).resolve()),
            "cord": str(Path(args.cord).resolve()),
        }
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootlets", required=True)
    parser.add_argument("--cord", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    print(json.dumps(run(get_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
