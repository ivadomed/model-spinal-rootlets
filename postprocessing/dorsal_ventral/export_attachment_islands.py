#!/usr/bin/env python3
"""Export numbered proximal attachment islands for lightweight expert review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from postprocessing.dorsal_ventral.split_rootlets import (
    _from_canonical,
    _interpolated_cord_centres,
    _local_surface_coordinates,
    _save_like,
    _to_canonical,
    _validate_inputs,
)


REVIEW_FIELDS = [
    "subject",
    "attachment_id",
    "level",
    "side",
    "component_id",
    "predicted_class",
    "median_ap_mm",
    "minimum_cord_distance_mm",
    "eligible_for_splitter",
    "voxel_count",
    "expert_class",
    "attachment_visible",
    "reviewer_confidence",
    "notes",
]


def extract_attachment_islands(
    rootlets: np.ndarray,
    cord: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    affine: np.ndarray | None = None,
    subject: str = "",
    attachment_distance_mm: float = 4.0,
    attachment_band_mm: float = 0.8,
    ap_margin_mm: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = _validate_inputs(rootlets, cord)
    if affine is None:
        affine = np.diag((*spacing, 1.0))
    affine = np.asarray(affine, dtype=float)
    support = labels > 0
    active = support | (cord > 0)
    active_slice = ndimage.find_objects(active.astype(np.uint8))[0]
    crop_transform = np.eye(4)
    crop_transform[:3, 3] = [axis_slice.start for axis_slice in active_slice]
    affine = affine @ crop_transform
    labels_crop = labels[active_slice]
    cord_crop = cord[active_slice] > 0

    centre_x, centre_y = _interpolated_cord_centres(cord_crop)
    distance_to_cord, nearest = ndimage.distance_transform_edt(
        ~cord_crop, sampling=spacing, return_indices=True
    )
    surface_right_mm, surface_ap_mm = _local_surface_coordinates(
        nearest, centre_x, centre_y, affine
    )

    island_map_crop = np.zeros(labels_crop.shape, dtype=np.int32)
    records: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    next_id = 1
    for level in np.unique(labels_crop[labels_crop > 0]):
        level_mask = labels_crop == level
        for side, side_mask in (
            ("right", level_mask & (surface_right_mm >= 0)),
            ("left", level_mask & (surface_right_mm < 0)),
        ):
            components, count = ndimage.label(side_mask, structure=structure)
            for component_id, component_slice in enumerate(
                ndimage.find_objects(components), start=1
            ):
                if component_slice is None:
                    continue
                component = components[component_slice] == component_id
                local_distance = distance_to_cord[component_slice]
                local_ap = surface_ap_mm[component_slice]
                minimum_distance = float(np.min(local_distance[component]))
                eligible = minimum_distance <= attachment_distance_mm
                band_limit = minimum_distance + attachment_band_mm
                if eligible:
                    band_limit = min(band_limit, attachment_distance_mm)
                band = component & (local_distance <= band_limit)
                islands, island_count = ndimage.label(band, structure=structure)
                for local_id in range(1, island_count + 1):
                    island = islands == local_id
                    median_ap = float(np.median(local_ap[island]))
                    if median_ap <= -ap_margin_mm:
                        predicted_class = "dorsal"
                    elif median_ap >= ap_margin_mm:
                        predicted_class = "ventral"
                    else:
                        predicted_class = "unclear"
                    map_view = island_map_crop[component_slice]
                    map_view[island] = next_id
                    records.append(
                        {
                            "subject": subject,
                            "attachment_id": next_id,
                            "level": int(level),
                            "side": side,
                            "component_id": int(component_id),
                            "predicted_class": predicted_class,
                            "median_ap_mm": round(median_ap, 4),
                            "minimum_cord_distance_mm": round(minimum_distance, 4),
                            "eligible_for_splitter": str(eligible).lower(),
                            "voxel_count": int(np.count_nonzero(island)),
                            "expert_class": "",
                            "attachment_visible": "",
                            "reviewer_confidence": "",
                            "notes": "",
                        }
                    )
                    next_id += 1

    island_map = np.zeros(labels.shape, dtype=np.int32)
    island_map[active_slice] = island_map_crop
    return island_map, records


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--cord", required=True)
    parser.add_argument("--subject", default="")
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--attachment-distance-mm", type=float, default=4.0)
    parser.add_argument("--attachment-band-mm", type=float, default=0.8)
    parser.add_argument("--ap-margin-mm", type=float, default=0.5)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    rootlet_image = nib.load(args.combined)
    cord_image = nib.load(args.cord)
    if rootlet_image.shape != cord_image.shape or not np.allclose(
        rootlet_image.affine, cord_image.affine, atol=1e-4
    ):
        raise ValueError("Combined rootlets and cord must use the same grid.")
    rootlets_ras, transform = _to_canonical(
        np.asanyarray(rootlet_image.dataobj), rootlet_image.affine
    )
    cord_ras, cord_transform = _to_canonical(
        np.asanyarray(cord_image.dataobj), cord_image.affine
    )
    if not np.array_equal(transform, cord_transform):
        raise ValueError("Combined rootlets and cord orientation transforms differ.")
    canonical_affine = rootlet_image.affine @ nib.orientations.inv_ornt_aff(
        transform, rootlet_image.shape
    )
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(canonical_affine))
    island_map, records = extract_attachment_islands(
        rootlets_ras,
        cord_ras,
        spacing,
        affine=canonical_affine,
        subject=args.subject,
        attachment_distance_mm=args.attachment_distance_mm,
        attachment_band_mm=args.attachment_band_mm,
        ap_margin_mm=args.ap_margin_mm,
    )
    island_native = _from_canonical(island_map, rootlet_image.affine)
    _save_like(island_native, rootlet_image, Path(args.output_map), np.int32)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported {len(records)} attachment islands to {csv_path}")


if __name__ == "__main__":
    main()
