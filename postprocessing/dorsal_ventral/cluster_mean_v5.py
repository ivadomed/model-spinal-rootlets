#!/usr/bin/env python3
"""Deterministic cluster-count and mean-AP gate for dorsal/ventral rootlets.

V5 operates only on the level-labelled RootletSeg support.  It reorients the
input to RPI, finds 3-D connected components independently at each spinal
level, and compares the mean RPI Y coordinate of substantial components.
Lower RPI Y is anterior/ventral; higher RPI Y is posterior/dorsal.

The method intentionally abstains at ambiguous levels.  A downstream
voxelwise fallback must fill those voxels; V5 never presents a forced guess as
a confident deterministic decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


RPI_ORIENTATION = nib.orientations.axcodes2ornt(("R", "P", "I"))


@dataclass(frozen=True)
class ComponentSummary:
    """Geometry used by the V5 gate for one connected component."""

    component_id: int
    voxel_count: int
    mean_y_mm: float
    minimum_y_mm: float
    maximum_y_mm: float


@dataclass(frozen=True)
class ClusterMeanV5Result:
    """Support partition and quality-control record returned by V5."""

    dorsal: np.ndarray
    ventral: np.ndarray
    fallback: np.ndarray
    qc: dict[str, Any]


def _validated_labels(rootlets_rpi: np.ndarray) -> np.ndarray:
    if rootlets_rpi.ndim != 3:
        raise ValueError("Rootlet labels must be a 3-D volume.")
    if not np.all(np.isfinite(rootlets_rpi)):
        raise ValueError("Rootlet labels must not contain NaN or infinite values.")
    rounded = np.rint(rootlets_rpi)
    if not np.allclose(rootlets_rpi, rounded, atol=1e-4):
        raise ValueError("Rootlet labels must contain integer values.")
    labels = rounded.astype(np.int32)
    if np.any(labels < 0):
        raise ValueError("Rootlet labels must be non-negative.")
    if not np.any(labels > 0):
        raise ValueError("Rootlet labels are empty.")
    return labels


def _component_summaries(
    component_map: np.ndarray,
    component_count: int,
    *,
    spacing_y_mm: float,
) -> list[ComponentSummary]:
    summaries: list[ComponentSummary] = []
    for component_id in range(1, component_count + 1):
        coordinates = np.argwhere(component_map == component_id)
        y_mm = coordinates[:, 1].astype(float) * spacing_y_mm
        summaries.append(
            ComponentSummary(
                component_id=component_id,
                voxel_count=int(len(coordinates)),
                mean_y_mm=float(np.mean(y_mm)),
                minimum_y_mm=float(np.min(y_mm)),
                maximum_y_mm=float(np.max(y_mm)),
            )
        )
    return summaries


def _mean_y_boundary(
    components: list[ComponentSummary],
    *,
    min_component_voxels: int,
    min_gap_mm: float,
    min_gap_ratio: float,
    allow_two_components: bool,
) -> tuple[float | None, str, list[ComponentSummary], dict[str, Any]]:
    substantial = [
        component
        for component in components
        if component.voxel_count >= min_component_voxels
    ]
    count = len(substantial)
    allowed_counts = {3, 4}
    if allow_two_components:
        allowed_counts.add(2)
    details: dict[str, Any] = {
        "substantial_components": count,
        "substantial_component_ids": [item.component_id for item in substantial],
        "ordered_mean_y_mm": [],
        "gaps_mm": [],
        "largest_gap_mm": None,
        "gap_ratio": None,
        "split_index": None,
    }
    if count not in allowed_counts:
        return None, "component_count", substantial, details

    ordered = sorted(substantial, key=lambda item: item.mean_y_mm)
    means = np.asarray([item.mean_y_mm for item in ordered], dtype=float)
    gaps = np.diff(means)
    split_index = int(np.argmax(gaps)) + 1
    largest_gap = float(gaps[split_index - 1])
    competing_gaps = np.delete(gaps, split_index - 1)
    second_gap = float(np.max(competing_gaps)) if len(competing_gaps) else 0.0
    gap_ratio = (
        float("inf") if second_gap <= np.finfo(float).eps else largest_gap / second_gap
    )
    details.update(
        {
            "ordered_mean_y_mm": [float(value) for value in means],
            "gaps_mm": [float(value) for value in gaps],
            "largest_gap_mm": largest_gap,
            "gap_ratio": gap_ratio,
            "split_index": split_index,
        }
    )

    if count == 4 and split_index != 2:
        return None, "unbalanced_four", substantial, details
    if largest_gap < min_gap_mm:
        return None, "weak_gap", substantial, details
    if count > 2 and gap_ratio < min_gap_ratio:
        return None, "non_dominant_gap", substantial, details

    boundary = float((means[split_index - 1] + means[split_index]) / 2.0)
    return boundary, "deterministic", substantial, details


def split_cluster_mean_v5(
    rootlets_rpi: np.ndarray,
    spacing_y_mm: float,
    *,
    min_component_voxels: int = 20,
    min_gap_mm: float = 0.5,
    min_gap_ratio: float = 1.25,
    allow_two_components: bool = True,
    crossing_fraction: float = 0.15,
) -> ClusterMeanV5Result:
    """Partition confident levels and route ambiguous levels to a fallback.

    The input must already be represented in RPI voxel order.  Output arrays
    retain the input level values.  Their sum exactly equals the input labels.
    """

    labels = _validated_labels(rootlets_rpi)
    if spacing_y_mm <= 0:
        raise ValueError("spacing_y_mm must be positive.")
    if min_component_voxels <= 0:
        raise ValueError("min_component_voxels must be positive.")
    if min_gap_mm < 0:
        raise ValueError("min_gap_mm must be non-negative.")
    if min_gap_ratio < 1:
        raise ValueError("min_gap_ratio must be at least 1.")
    if not 0 <= crossing_fraction < 0.5:
        raise ValueError("crossing_fraction must be in [0, 0.5).")

    dorsal = np.zeros(labels.shape, dtype=np.int32)
    ventral = np.zeros(labels.shape, dtype=np.int32)
    fallback = np.zeros(labels.shape, dtype=np.int32)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    level_records: list[dict[str, Any]] = []

    for level_value in np.unique(labels[labels > 0]):
        level = int(level_value)
        level_mask = labels == level
        component_map, component_count = ndimage.label(level_mask, structure=structure)
        components = _component_summaries(
            component_map,
            int(component_count),
            spacing_y_mm=spacing_y_mm,
        )
        boundary, decision, substantial, details = _mean_y_boundary(
            components,
            min_component_voxels=min_component_voxels,
            min_gap_mm=min_gap_mm,
            min_gap_ratio=min_gap_ratio,
            allow_two_components=allow_two_components,
        )

        crossing_ids: list[int] = []
        if boundary is not None:
            for component in substantial:
                component_y = (
                    np.argwhere(component_map == component.component_id)[:, 1]
                    * spacing_y_mm
                )
                fraction_ventral = float(np.mean(component_y <= boundary))
                fraction_dorsal = float(np.mean(component_y > boundary))
                if (
                    fraction_ventral >= crossing_fraction
                    and fraction_dorsal >= crossing_fraction
                ):
                    crossing_ids.append(component.component_id)
            if crossing_ids:
                boundary = None
                decision = "boundary_crossing_component"

        if boundary is None:
            fallback[level_mask] = level
        else:
            for component in components:
                support = component_map == component.component_id
                if component.mean_y_mm <= boundary:
                    ventral[support] = level
                else:
                    dorsal[support] = level

        level_records.append(
            {
                "level": level,
                "component_count": int(component_count),
                "component_voxel_counts": [item.voxel_count for item in components],
                "component_mean_y_mm": [item.mean_y_mm for item in components],
                "decision": decision,
                "boundary_y_mm": boundary,
                "boundary_crossing_component_ids": crossing_ids,
                **details,
            }
        )

    if np.any((dorsal > 0) & (ventral > 0)):
        raise RuntimeError("Internal error: dorsal and ventral outputs overlap.")
    if np.any(((dorsal > 0) | (ventral > 0)) & (fallback > 0)):
        raise RuntimeError("Internal error: deterministic and fallback outputs overlap.")
    if not np.array_equal(dorsal + ventral + fallback, labels):
        raise RuntimeError("Internal error: V5 changed the RootletSeg support or levels.")

    support_voxels = int(np.count_nonzero(labels))
    deterministic_voxels = int(np.count_nonzero((dorsal > 0) | (ventral > 0)))
    deterministic_levels = sum(
        record["decision"] == "deterministic" for record in level_records
    )
    qc = {
        "method": "cluster_count_mean_rpi_y_v5",
        "orientation": "RPI",
        "rule": "lower mean RPI Y is ventral; higher mean RPI Y is dorsal",
        "min_component_voxels": int(min_component_voxels),
        "min_gap_mm": float(min_gap_mm),
        "min_gap_ratio": float(min_gap_ratio),
        "allow_two_components": bool(allow_two_components),
        "crossing_fraction": float(crossing_fraction),
        "input_voxels": support_voxels,
        "deterministic_voxels": deterministic_voxels,
        "fallback_voxels": int(np.count_nonzero(fallback)),
        "deterministic_voxel_coverage": deterministic_voxels / support_voxels,
        "deterministic_levels": int(deterministic_levels),
        "fallback_levels": int(len(level_records) - deterministic_levels),
        "levels": level_records,
    }
    return ClusterMeanV5Result(dorsal, ventral, fallback, qc)


def _to_rpi(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(original, RPI_ORIENTATION)
    return nib.orientations.apply_orientation(data, transform), transform


def _from_rpi(data: np.ndarray, original_affine: np.ndarray) -> np.ndarray:
    original = nib.orientations.io_orientation(original_affine)
    transform = nib.orientations.ornt_transform(RPI_ORIENTATION, original)
    return nib.orientations.apply_orientation(data, transform)


def _save_like(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.int16)
    image = nib.Nifti1Image(data.astype(np.int16), reference.affine, header=header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rootlet_image = nib.load(args.rootlets)
    rootlets_rpi, transform = _to_rpi(
        np.asanyarray(rootlet_image.dataobj), rootlet_image.affine
    )
    rpi_affine = rootlet_image.affine @ nib.orientations.inv_ornt_aff(
        transform, rootlet_image.shape
    )
    spacing_y_mm = float(nib.affines.voxel_sizes(rpi_affine)[1])
    result = split_cluster_mean_v5(
        rootlets_rpi,
        spacing_y_mm,
        min_component_voxels=args.min_component_voxels,
        min_gap_mm=args.min_gap_mm,
        min_gap_ratio=args.min_gap_ratio,
        allow_two_components=not args.reject_two_components,
        crossing_fraction=args.crossing_fraction,
    )
    _save_like(
        _from_rpi(result.dorsal, rootlet_image.affine),
        rootlet_image,
        Path(args.output_dorsal),
    )
    _save_like(
        _from_rpi(result.ventral, rootlet_image.affine),
        rootlet_image,
        Path(args.output_ventral),
    )
    _save_like(
        _from_rpi(result.fallback, rootlet_image.affine),
        rootlet_image,
        Path(args.output_fallback),
    )
    qc = dict(result.qc)
    qc.update(
        {
            "rootlets": str(Path(args.rootlets).resolve()),
            "output_dorsal": str(Path(args.output_dorsal).resolve()),
            "output_ventral": str(Path(args.output_ventral).resolve()),
            "output_fallback": str(Path(args.output_fallback).resolve()),
        }
    )
    if args.qc_json:
        qc_path = Path(args.qc_json)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(qc, indent=2, allow_nan=False) + "\n")
    return qc


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootlets", required=True)
    parser.add_argument("--output-dorsal", required=True)
    parser.add_argument("--output-ventral", required=True)
    parser.add_argument("--output-fallback", required=True)
    parser.add_argument("--qc-json")
    parser.add_argument("--min-component-voxels", type=int, default=20)
    parser.add_argument("--min-gap-mm", type=float, default=0.5)
    parser.add_argument("--min-gap-ratio", type=float, default=1.25)
    parser.add_argument("--crossing-fraction", type=float, default=0.15)
    parser.add_argument("--reject-two-components", action="store_true")
    return parser


def main() -> None:
    qc = run(get_parser().parse_args())
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
