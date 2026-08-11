#!/usr/bin/env python3
"""Audit deterministic attachment seeds using dorsal-only positive references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from postprocessing.dorsal_ventral.split_rootlets import (
    _interpolated_cord_centres,
    _local_surface_coordinates,
    _to_canonical,
    _validate_inputs,
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _validate_reference(reference: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if reference.shape != shape:
        raise ValueError("Dorsal reference shape differs from the combined mask.")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Dorsal reference contains NaN or infinite values.")
    rounded = np.rint(reference)
    if not np.allclose(reference, rounded, atol=1e-4):
        raise ValueError("Dorsal reference must be integer-valued.")
    if np.any(rounded < 0):
        raise ValueError("Dorsal reference must be non-negative.")
    return rounded.astype(np.int32)


def audit_attachment_seeds(
    rootlets: np.ndarray,
    cord: np.ndarray,
    dorsal_reference: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    affine: np.ndarray | None = None,
    attachment_distance_mm: float = 4.0,
    ap_margin_mm: float = 0.5,
) -> dict[str, Any]:
    """Measure whether known dorsal positives receive dorsal or ventral seeds."""
    labels = _validate_inputs(rootlets, cord)
    reference = _validate_reference(dorsal_reference, labels.shape)
    if affine is None:
        affine = np.diag((*spacing, 1.0))
    affine = np.asarray(affine, dtype=float)

    reference_counts = {
        int(label): int(np.count_nonzero(reference == label))
        for label in np.unique(reference[reference > 0])
    }
    active = (labels > 0) | (cord > 0)
    active_slice = ndimage.find_objects(active.astype(np.uint8))[0]
    crop_transform = np.eye(4)
    crop_transform[:3, 3] = [axis_slice.start for axis_slice in active_slice]
    affine = affine @ crop_transform
    labels = labels[active_slice]
    cord_mask = cord[active_slice] > 0
    reference = reference[active_slice]

    centre_x, centre_y = _interpolated_cord_centres(cord_mask)
    distance_to_cord, nearest = ndimage.distance_transform_edt(
        ~cord_mask, sampling=spacing, return_indices=True
    )
    surface_right_mm, surface_ap_mm = _local_surface_coordinates(
        nearest, centre_x, centre_y, affine
    )

    support = labels > 0
    proximal = support & (distance_to_cord <= attachment_distance_mm)
    dorsal_seed = proximal & (surface_ap_mm <= -ap_margin_mm)
    ventral_seed = proximal & (surface_ap_mm >= ap_margin_mm)
    neutral_seed = proximal & ~dorsal_seed & ~ventral_seed
    known_dorsal = (reference > 0) & support
    known_proximal = known_dorsal & proximal

    def seed_metrics(mask: np.ndarray, reference_total: int) -> dict[str, Any]:
        common = known_dorsal & mask
        proximal_common = known_proximal & mask
        common_count = int(np.count_nonzero(common))
        proximal_count = int(np.count_nonzero(proximal_common))
        dorsal_count = int(np.count_nonzero(proximal_common & dorsal_seed))
        ventral_count = int(np.count_nonzero(proximal_common & ventral_seed))
        neutral_count = int(np.count_nonzero(proximal_common & neutral_seed))
        return {
            "reference_voxels": int(reference_total),
            "reference_voxels_in_combined_support": common_count,
            "reference_coverage": _ratio(common_count, reference_total),
            "known_dorsal_proximal_voxels": proximal_count,
            "known_dorsal_proximal_coverage": _ratio(proximal_count, common_count),
            "known_dorsal_dorsal_seed_rate": _ratio(dorsal_count, proximal_count),
            "known_dorsal_ventral_seed_rate": _ratio(ventral_count, proximal_count),
            "known_dorsal_neutral_seed_rate": _ratio(neutral_count, proximal_count),
        }

    result = seed_metrics(
        np.ones(labels.shape, dtype=bool), sum(reference_counts.values())
    )
    per_label = []
    for label, reference_total in sorted(reference_counts.items()):
        record = seed_metrics(reference == label, reference_total)
        record["label"] = label
        per_label.append(record)

    component_states = {
        state: {"components": 0, "known_dorsal_voxels": 0}
        for state in ("dorsal_only", "ventral_only", "dual_seed", "unseeded")
    }
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    for level in np.unique(labels[support]):
        level_mask = labels == level
        for side_mask in (
            level_mask & (surface_right_mm >= 0),
            level_mask & (surface_right_mm < 0),
        ):
            components, count = ndimage.label(side_mask, structure=structure)
            for component_id in range(1, count + 1):
                component = components == component_id
                known_count = int(np.count_nonzero(component & known_dorsal))
                if not known_count:
                    continue
                has_dorsal = bool(np.any(component & dorsal_seed))
                has_ventral = bool(np.any(component & ventral_seed))
                if has_dorsal and has_ventral:
                    state = "dual_seed"
                elif has_dorsal:
                    state = "dorsal_only"
                elif has_ventral:
                    state = "ventral_only"
                else:
                    state = "unseeded"
                component_states[state]["components"] += 1
                component_states[state]["known_dorsal_voxels"] += known_count

    result.update(
        {
            "method": "proximal_attachment_seed_audit_v1",
            "attachment_distance_mm": float(attachment_distance_mm),
            "ap_margin_mm": float(ap_margin_mm),
            "labels": per_label,
            "known_dorsal_component_seed_states": component_states,
            "limitations": [
                "Dorsal-only reference; no ventral seed accuracy is identifiable.",
                "Voxel seed rates do not establish anatomical branch identity.",
            ],
        }
    )
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--cord", required=True)
    parser.add_argument("--dorsal-reference", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--attachment-distance-mm", type=float, default=4.0)
    parser.add_argument("--ap-margin-mm", type=float, default=0.5)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    paths = [args.combined, args.cord, args.dorsal_reference]
    images = [nib.load(path) for path in paths]
    for image, path in zip(images[1:], paths[1:]):
        if image.shape != images[0].shape or not np.allclose(
            image.affine, images[0].affine, atol=1e-4
        ):
            raise ValueError(f"Image is not on the combined-mask grid: {path}")

    arrays_and_transforms = [
        _to_canonical(np.asanyarray(image.dataobj), image.affine) for image in images
    ]
    transforms = [item[1] for item in arrays_and_transforms]
    if not all(np.array_equal(transforms[0], transform) for transform in transforms[1:]):
        raise ValueError("Input orientation transforms differ.")
    canonical_affine = images[0].affine @ nib.orientations.inv_ornt_aff(
        transforms[0], images[0].shape
    )
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(canonical_affine))
    metrics = audit_attachment_seeds(
        *(item[0] for item in arrays_and_transforms),
        spacing,
        affine=canonical_affine,
        attachment_distance_mm=args.attachment_distance_mm,
        ap_margin_mm=args.ap_margin_mm,
    )
    metrics["inputs"] = {
        "combined": str(Path(args.combined).resolve()),
        "cord": str(Path(args.cord).resolve()),
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
