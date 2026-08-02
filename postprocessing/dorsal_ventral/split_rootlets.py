#!/usr/bin/env python3
"""Split a level-labelled RootletSeg mask into dorsal and ventral masks.

The separator is anatomy-first and support-constrained. It identifies proximal
rootlet voxels near the spinal cord, labels their attachment as posterior
(dorsal) or anterior (ventral), and propagates those identities geodesically
inside each connected component. The input rootlet support and level values are
preserved exactly between the two outputs.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


RAS_ORIENTATION = nib.orientations.axcodes2ornt(("R", "A", "S"))
SEED_STRATEGIES = ("attachment_island", "dense_voxel")


@dataclass(frozen=True)
class SplitResult:
    """Arrays and quality-control information produced by the separator."""

    dorsal: np.ndarray
    ventral: np.ndarray
    score: np.ndarray
    qc: dict[str, Any]


def _validate_inputs(rootlets: np.ndarray, cord: np.ndarray) -> np.ndarray:
    if rootlets.ndim != 3 or cord.ndim != 3:
        raise ValueError("Rootlet and spinal cord inputs must both be 3D.")
    if rootlets.shape != cord.shape:
        raise ValueError(
            f"Rootlet and spinal cord shapes differ: {rootlets.shape} != {cord.shape}."
        )
    if not np.all(np.isfinite(rootlets)) or not np.all(np.isfinite(cord)):
        raise ValueError("Inputs must not contain NaN or infinite values.")
    rounded = np.rint(rootlets)
    if not np.allclose(rootlets, rounded, atol=1e-4):
        raise ValueError("Rootlet labels must be integer-valued.")
    labels = rounded.astype(np.int32)
    if np.any(labels < 0):
        raise ValueError("Rootlet labels must be non-negative.")
    if not np.any(cord > 0):
        raise ValueError("The spinal cord mask is empty.")
    return labels


def _interpolated_cord_centres(cord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed x/y cord centroids for every canonical RAS z slice."""
    centre_x = np.full(cord.shape[2], np.nan, dtype=float)
    centre_y = np.full(cord.shape[2], np.nan, dtype=float)
    for z in np.flatnonzero(np.any(cord, axis=(0, 1))):
        x, y = np.nonzero(cord[:, :, z])
        centre_x[z] = np.mean(x)
        centre_y[z] = np.mean(y)

    valid = np.flatnonzero(np.isfinite(centre_x))
    grid = np.arange(cord.shape[2])
    centre_x = np.interp(grid, valid, centre_x[valid])
    centre_y = np.interp(grid, valid, centre_y[valid])

    # A short superior-inferior smoothing suppresses single-slice mask noise.
    sigma = min(2.0, max(0.0, len(valid) / 20.0))
    if sigma > 0:
        centre_x = ndimage.gaussian_filter1d(centre_x, sigma=sigma, mode="nearest")
        centre_y = ndimage.gaussian_filter1d(centre_y, sigma=sigma, mode="nearest")
    return centre_x, centre_y


def _local_surface_coordinates(
    nearest: np.ndarray,
    centre_x: np.ndarray,
    centre_y: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return right/left and posterior/anterior surface coordinates in mm.

    The global RAS anterior axis is projected onto the plane normal to the local
    cord centreline tangent. This retains anatomical meaning for oblique and
    curved cords instead of treating canonical voxel y as the AP direction.
    """
    z_grid = np.arange(len(centre_x), dtype=float)
    centre_voxels = np.column_stack((centre_x, centre_y, z_grid))
    centre_world = nib.affines.apply_affine(affine, centre_voxels)

    tangent = np.gradient(centre_world, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(tangent_norm, np.finfo(float).eps)

    global_anterior = np.array([0.0, 1.0, 0.0])
    local_ap = global_anterior - tangent * (tangent @ global_anterior)[:, None]
    ap_norm = np.linalg.norm(local_ap, axis=1, keepdims=True)
    degenerate = ap_norm[:, 0] < 1e-6
    local_ap /= np.maximum(ap_norm, np.finfo(float).eps)
    local_ap[degenerate] = global_anterior

    local_right = np.cross(local_ap, tangent)
    right_norm = np.linalg.norm(local_right, axis=1, keepdims=True)
    local_right /= np.maximum(right_norm, np.finfo(float).eps)
    local_right[local_right[:, 0] < 0] *= -1

    nearest_x, nearest_y, nearest_z = nearest
    surface_world = np.empty(nearest.shape[1:] + (3,), dtype=np.float32)
    for world_axis in range(3):
        surface_world[..., world_axis] = (
            affine[world_axis, 0] * nearest_x
            + affine[world_axis, 1] * nearest_y
            + affine[world_axis, 2] * nearest_z
            + affine[world_axis, 3]
        )
    delta = surface_world - centre_world[nearest_z]
    surface_right_mm = np.sum(delta * local_right[nearest_z], axis=-1)
    surface_ap_mm = np.sum(delta * local_ap[nearest_z], axis=-1)
    return surface_right_mm, surface_ap_mm


def _neighbour_offsets(
    spacing: tuple[float, float, float],
) -> list[tuple[tuple[int, int, int], float]]:
    neighbours = []
    spacing_array = np.asarray(spacing, dtype=float)
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        weight = float(np.linalg.norm(np.asarray(offset) * spacing_array))
        neighbours.append((offset, weight))
    return neighbours


def _geodesic_distance(
    support: np.ndarray,
    seeds: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Multi-source Dijkstra distance constrained to a small binary support."""
    distance = np.full(support.shape, np.inf, dtype=np.float32)
    queue: list[tuple[float, int, int, int]] = []
    for x, y, z in np.argwhere(seeds):
        distance[x, y, z] = 0.0
        heapq.heappush(queue, (0.0, int(x), int(y), int(z)))

    neighbours = _neighbour_offsets(spacing)
    shape = support.shape
    while queue:
        current, x, y, z = heapq.heappop(queue)
        if current > float(distance[x, y, z]) + 1e-6:
            continue
        for (dx, dy, dz), edge_length in neighbours:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue
            if not support[nx, ny, nz]:
                continue
            candidate = current + edge_length
            if candidate < float(distance[nx, ny, nz]):
                distance[nx, ny, nz] = candidate
                heapq.heappush(queue, (candidate, nx, ny, nz))
    return distance


def _component_seeds(
    component: np.ndarray,
    distance_to_cord: np.ndarray,
    surface_ap_mm: np.ndarray,
    structure: np.ndarray,
    *,
    strategy: str,
    attachment_distance_mm: float,
    attachment_band_mm: float,
    ap_margin_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Construct dorsal/ventral seeds inside one cropped component."""
    proximal = component & (distance_to_cord <= attachment_distance_mm)
    if strategy == "dense_voxel":
        dorsal = proximal & (surface_ap_mm <= -ap_margin_mm)
        ventral = proximal & (surface_ap_mm >= ap_margin_mm)
        return dorsal, ventral, {"attachment_islands": 0, "neutral_islands": 0}

    dorsal = np.zeros(component.shape, dtype=bool)
    ventral = np.zeros(component.shape, dtype=bool)
    if not np.any(proximal):
        return dorsal, ventral, {"attachment_islands": 0, "neutral_islands": 0}

    minimum_distance = float(np.min(distance_to_cord[component]))
    band_limit = min(
        attachment_distance_mm, minimum_distance + attachment_band_mm
    )
    attachment_band = component & (distance_to_cord <= band_limit)
    islands, count = ndimage.label(attachment_band, structure=structure)
    neutral_count = 0
    for island_id in range(1, count + 1):
        island = islands == island_id
        median_ap = float(np.median(surface_ap_mm[island]))
        if median_ap <= -ap_margin_mm:
            dorsal[island] = True
        elif median_ap >= ap_margin_mm:
            ventral[island] = True
        else:
            neutral_count += 1
    return dorsal, ventral, {
        "attachment_islands": int(count),
        "neutral_islands": int(neutral_count),
    }


def split_rootlets(
    rootlets: np.ndarray,
    cord: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    affine: np.ndarray | None = None,
    seed_strategy: str = "attachment_island",
    attachment_distance_mm: float = 4.0,
    attachment_band_mm: float = 0.8,
    ap_margin_mm: float = 0.5,
) -> SplitResult:
    """Split arrays represented in canonical orientation with a RAS affine.

    The world anterior axis is projected normal to the local cord centreline.
    Dorsal seeds have a negative coordinate in that frame; ventral seeds have a
    positive coordinate.
    """
    labels = _validate_inputs(rootlets, cord)
    if any(value <= 0 for value in spacing):
        raise ValueError(f"Voxel spacing must be positive, got {spacing}.")
    if attachment_distance_mm <= 0:
        raise ValueError("attachment_distance_mm must be positive.")
    if attachment_band_mm <= 0:
        raise ValueError("attachment_band_mm must be positive.")
    if ap_margin_mm < 0:
        raise ValueError("ap_margin_mm must be non-negative.")
    if seed_strategy not in SEED_STRATEGIES:
        raise ValueError(
            f"seed_strategy must be one of {SEED_STRATEGIES}, got {seed_strategy!r}."
        )

    if affine is None:
        affine = np.diag((*spacing, 1.0))
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("affine must be a finite 4x4 matrix.")

    support = labels > 0
    cord_mask = cord > 0
    active = support | cord_mask
    active_slice = ndimage.find_objects(active.astype(np.uint8))[0]
    full_slice = tuple(slice(0, size) for size in labels.shape)
    if active_slice != full_slice:
        crop_transform = np.eye(4)
        crop_transform[:3, 3] = [axis_slice.start for axis_slice in active_slice]
        cropped = split_rootlets(
            labels[active_slice],
            cord_mask[active_slice],
            spacing,
            affine=affine @ crop_transform,
            seed_strategy=seed_strategy,
            attachment_distance_mm=attachment_distance_mm,
            attachment_band_mm=attachment_band_mm,
            ap_margin_mm=ap_margin_mm,
        )
        dorsal_full = np.zeros(labels.shape, dtype=np.int32)
        ventral_full = np.zeros(labels.shape, dtype=np.int32)
        score_full = np.zeros(labels.shape, dtype=np.float32)
        dorsal_full[active_slice] = cropped.dorsal
        ventral_full[active_slice] = cropped.ventral
        score_full[active_slice] = cropped.score
        qc = dict(cropped.qc)
        qc["input_shape"] = list(labels.shape)
        qc["processing_bbox"] = [
            [int(axis_slice.start), int(axis_slice.stop)] for axis_slice in active_slice
        ]
        return SplitResult(dorsal_full, ventral_full, score_full, qc)

    centre_x, centre_y = _interpolated_cord_centres(cord_mask)
    distance_to_cord, nearest = ndimage.distance_transform_edt(
        ~cord_mask,
        sampling=spacing,
        return_indices=True,
    )

    surface_right_mm, surface_ap_mm = _local_surface_coordinates(
        nearest, centre_x, centre_y, affine
    )

    dorsal = np.zeros(labels.shape, dtype=np.int32)
    ventral = np.zeros(labels.shape, dtype=np.int32)
    score = np.zeros(labels.shape, dtype=np.float32)
    level_records: list[dict[str, Any]] = []
    total_fallbacks = 0
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    for level in np.unique(labels[support]):
        level_mask = labels == level
        level_record: dict[str, Any] = {
            "label": int(level),
            "voxels": int(np.count_nonzero(level_mask)),
            "components": 0,
            "geodesic_components": 0,
            "single_seed_class_components": 0,
            "fallback_components": 0,
            "attachment_islands": 0,
            "neutral_attachment_islands": 0,
        }

        # Keep left and right propagation separate even if a prediction creates
        # an anatomically implausible bridge across the midline.
        side_masks = {
            "right": level_mask & (surface_right_mm >= 0),
            "left": level_mask & (surface_right_mm < 0),
        }
        for _side_name, side_mask in side_masks.items():
            components, count = ndimage.label(side_mask, structure=structure)
            level_record["components"] += int(count)
            for component_id, component_slice in enumerate(
                ndimage.find_objects(components), start=1
            ):
                if component_slice is None:
                    continue
                component = components[component_slice] == component_id
                local_distance = distance_to_cord[component_slice]
                local_ap = surface_ap_mm[component_slice]
                dorsal_seeds, ventral_seeds, seed_record = _component_seeds(
                    component,
                    local_distance,
                    local_ap,
                    structure,
                    strategy=seed_strategy,
                    attachment_distance_mm=attachment_distance_mm,
                    attachment_band_mm=attachment_band_mm,
                    ap_margin_mm=ap_margin_mm,
                )
                level_record["attachment_islands"] += seed_record[
                    "attachment_islands"
                ]
                level_record["neutral_attachment_islands"] += seed_record[
                    "neutral_islands"
                ]
                has_dorsal = bool(np.any(dorsal_seeds))
                has_ventral = bool(np.any(ventral_seeds))

                if has_dorsal and has_ventral:
                    d_dorsal = _geodesic_distance(component, dorsal_seeds, spacing)
                    d_ventral = _geodesic_distance(component, ventral_seeds, spacing)
                    dorsal_part = component & (d_dorsal <= d_ventral)
                    ventral_part = component & ~dorsal_part
                    component_score = np.zeros(component.shape, dtype=np.float32)
                    denominator = (
                        d_dorsal[component]
                        + d_ventral[component]
                        + np.finfo(np.float32).eps
                    )
                    component_score[component] = (
                        np.abs(d_dorsal[component] - d_ventral[component]) / denominator
                    )
                    score_view = score[component_slice]
                    score_view[component] = component_score[component]
                    level_record["geodesic_components"] += 1
                elif has_dorsal or has_ventral:
                    dorsal_part = component if has_dorsal else np.zeros_like(component)
                    ventral_part = component if has_ventral else np.zeros_like(component)
                    attachment_values = np.abs(
                        local_ap[dorsal_seeds | ventral_seeds]
                    )
                    seed_score = float(
                        np.clip(
                            np.median(attachment_values)
                            / max(ap_margin_mm * 4.0, 1.0),
                            0.0,
                            1.0,
                        )
                    )
                    score_view = score[component_slice]
                    score_view[component] = seed_score
                    level_record["single_seed_class_components"] += 1
                else:
                    # Disconnected fragments without a visible attachment retain
                    # support, but receive an explicitly low heuristic score.
                    median_ap = float(np.median(local_ap[component]))
                    is_dorsal = median_ap < 0
                    dorsal_part = component if is_dorsal else np.zeros_like(component)
                    ventral_part = component if not is_dorsal else np.zeros_like(component)
                    score_view = score[component_slice]
                    score_view[component] = min(abs(median_ap) / 8.0, 0.25)
                    level_record["fallback_components"] += 1
                    total_fallbacks += 1

                dorsal_view = dorsal[component_slice]
                ventral_view = ventral[component_slice]
                dorsal_view[dorsal_part] = int(level)
                ventral_view[ventral_part] = int(level)

        level_record["dorsal_voxels"] = int(np.count_nonzero(dorsal == level))
        level_record["ventral_voxels"] = int(np.count_nonzero(ventral == level))
        level_records.append(level_record)

    if np.any((dorsal > 0) & (ventral > 0)):
        raise RuntimeError("Internal error: dorsal and ventral outputs overlap.")
    if not np.array_equal(dorsal + ventral, labels):
        raise RuntimeError("Internal error: output support or level labels changed.")

    qc = {
        "method": (
            "attachment_island_geodesic_v2"
            if seed_strategy == "attachment_island"
            else "proximal_attachment_geodesic_v1"
        ),
        "orientation": "RAS",
        "posterior_definition": "negative centerline-normal RAS AP coordinate",
        "attachment_distance_mm": float(attachment_distance_mm),
        "attachment_band_mm": float(attachment_band_mm),
        "ap_margin_mm": float(ap_margin_mm),
        "seed_strategy": seed_strategy,
        "input_voxels": int(np.count_nonzero(support)),
        "dorsal_voxels": int(np.count_nonzero(dorsal)),
        "ventral_voxels": int(np.count_nonzero(ventral)),
        "fallback_components": int(total_fallbacks),
        "mean_heuristic_score": float(np.mean(score[support])) if np.any(support) else 0.0,
        "levels": level_records,
    }
    return SplitResult(dorsal=dorsal, ventral=ventral, score=score, qc=qc)


def _to_canonical(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original_orientation = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(original_orientation, RAS_ORIENTATION)
    return nib.orientations.apply_orientation(data, transform), transform


def _from_canonical(data: np.ndarray, original_affine: np.ndarray) -> np.ndarray:
    original_orientation = nib.orientations.io_orientation(original_affine)
    transform = nib.orientations.ornt_transform(RAS_ORIENTATION, original_orientation)
    return nib.orientations.apply_orientation(data, transform)


def _save_like(data: np.ndarray, reference: nib.Nifti1Image, path: Path, dtype: np.dtype) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(data.astype(dtype), reference.affine, header=header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rootlet_image = nib.load(args.rootlets)
    cord_image = nib.load(args.cord)
    if rootlet_image.shape != cord_image.shape or not np.allclose(
        rootlet_image.affine, cord_image.affine, atol=1e-4
    ):
        raise ValueError("Rootlet and spinal cord images must use the same voxel grid and affine.")

    rootlet_data = np.asanyarray(rootlet_image.dataobj)
    cord_data = np.asanyarray(cord_image.dataobj)
    rootlet_ras, rootlet_transform = _to_canonical(rootlet_data, rootlet_image.affine)
    cord_ras, cord_transform = _to_canonical(cord_data, cord_image.affine)
    if not np.array_equal(rootlet_transform, cord_transform):
        raise ValueError("Rootlet and spinal cord orientation transforms differ.")

    canonical_affine = rootlet_image.affine @ nib.orientations.inv_ornt_aff(
        rootlet_transform, rootlet_image.shape
    )
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(canonical_affine))
    result = split_rootlets(
        rootlet_ras,
        cord_ras,
        spacing,
        affine=canonical_affine,
        seed_strategy=args.seed_strategy,
        attachment_distance_mm=args.attachment_distance_mm,
        attachment_band_mm=args.attachment_band_mm,
        ap_margin_mm=args.ap_margin_mm,
    )

    dorsal_native = _from_canonical(result.dorsal, rootlet_image.affine)
    ventral_native = _from_canonical(result.ventral, rootlet_image.affine)
    score_native = _from_canonical(result.score, rootlet_image.affine)
    _save_like(dorsal_native, rootlet_image, Path(args.output_dorsal), np.int16)
    _save_like(ventral_native, rootlet_image, Path(args.output_ventral), np.int16)
    if args.output_score:
        _save_like(score_native, rootlet_image, Path(args.output_score), np.float32)

    qc = dict(result.qc)
    qc.update(
        {
            "rootlets": str(Path(args.rootlets).resolve()),
            "cord": str(Path(args.cord).resolve()),
            "output_dorsal": str(Path(args.output_dorsal).resolve()),
            "output_ventral": str(Path(args.output_ventral).resolve()),
        }
    )
    if args.qc_json:
        qc_path = Path(args.qc_json)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(qc, indent=2) + "\n")
    return qc


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootlets", required=True, help="Combined level-labelled rootlet NIfTI.")
    parser.add_argument("--cord", required=True, help="Spinal cord segmentation on the same grid.")
    parser.add_argument(
        "--output-dorsal", required=True, help="Dorsal level-labelled output NIfTI."
    )
    parser.add_argument(
        "--output-ventral", required=True, help="Ventral level-labelled output NIfTI."
    )
    parser.add_argument(
        "--output-score", help="Optional uncalibrated voxelwise heuristic-score NIfTI."
    )
    parser.add_argument("--qc-json", help="Optional component-level QC JSON report.")
    parser.add_argument(
        "--seed-strategy", choices=SEED_STRATEGIES, default="attachment_island"
    )
    parser.add_argument("--attachment-distance-mm", type=float, default=4.0)
    parser.add_argument("--attachment-band-mm", type=float, default=0.8)
    parser.add_argument("--ap-margin-mm", type=float, default=0.5)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    qc = run(args)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
