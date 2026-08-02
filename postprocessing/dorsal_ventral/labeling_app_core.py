#!/usr/bin/env python3
"""Core data model for local expert labeling of RootletSeg clusters."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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


EXPERT_CLASSES = ("dorsal", "ventral", "mixed", "unclear")
REVIEW_FIELDS = (
    "case_id",
    "reviewer_id",
    "cluster_id",
    "cluster_key",
    "cluster_fingerprint",
    "level",
    "side",
    "component_id",
    "voxel_count",
    "slice_start_ras",
    "slice_stop_ras",
    "centroid_x_ras",
    "centroid_y_ras",
    "centroid_z_ras",
    "median_attachment_ap_mm",
    "minimum_cord_distance_mm",
    "suggested_class",
    "expert_class",
    "attachment_visible",
    "reviewer_confidence",
    "notes",
    "updated_at",
)


@dataclass
class AnnotationCase:
    """Loaded images, canonical arrays, and deterministic cluster records."""

    case_id: str
    anatomy_path: Path
    rootlets_path: Path
    cord_path: Path
    anatomy_image: nib.Nifti1Image
    rootlets_image: nib.Nifti1Image
    anatomy_ras: np.ndarray
    rootlets_ras: np.ndarray
    cord_ras: np.ndarray
    cluster_map_ras: np.ndarray
    records: list[dict[str, Any]]


def _same_grid(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return first.shape == second.shape and np.allclose(
        first.affine, second.affine, atol=1e-4
    )


def extract_rootlet_clusters(
    rootlets: np.ndarray,
    cord: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    affine: np.ndarray | None = None,
    case_id: str = "",
    attachment_distance_mm: float = 4.0,
    attachment_band_mm: float = 0.8,
    ap_margin_mm: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract full connected rootlet branches, separated by level and side."""
    labels = _validate_inputs(rootlets, cord)
    if affine is None:
        affine = np.diag((*spacing, 1.0))
    cord_mask = cord > 0
    centre_x, centre_y = _interpolated_cord_centres(cord_mask)
    distance_to_cord, nearest = ndimage.distance_transform_edt(
        ~cord_mask,
        sampling=spacing,
        return_indices=True,
    )
    surface_right_mm, surface_ap_mm = _local_surface_coordinates(
        nearest, centre_x, centre_y, np.asarray(affine, dtype=float)
    )

    cluster_map = np.zeros(labels.shape, dtype=np.int32)
    records: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    next_id = 1
    for level in np.unique(labels[labels > 0]):
        level_mask = labels == level
        for side, side_mask in (
            ("right", level_mask & (surface_right_mm >= 0)),
            ("left", level_mask & (surface_right_mm < 0)),
        ):
            components, _ = ndimage.label(side_mask, structure=structure)
            for component_id, component_slice in enumerate(
                ndimage.find_objects(components), start=1
            ):
                if component_slice is None:
                    continue
                component = components[component_slice] == component_id
                local_distance = distance_to_cord[component_slice]
                local_ap = surface_ap_mm[component_slice]
                minimum_distance = float(np.min(local_distance[component]))
                band_limit = minimum_distance + attachment_band_mm
                if minimum_distance <= attachment_distance_mm:
                    band_limit = min(band_limit, attachment_distance_mm)
                attachment = component & (local_distance <= band_limit)
                median_ap = float(np.median(local_ap[attachment]))
                if median_ap <= -ap_margin_mm:
                    suggestion = "dorsal"
                elif median_ap >= ap_margin_mm:
                    suggestion = "ventral"
                else:
                    suggestion = "unclear"

                map_view = cluster_map[component_slice]
                map_view[component] = next_id
                coordinates = np.argwhere(component)
                offset = np.array([axis.start for axis in component_slice])
                coordinates += offset
                centroid = np.mean(coordinates, axis=0)
                z_values = coordinates[:, 2]
                cluster_key = f"L{int(level):02d}-{side}-C{component_id:02d}"
                fingerprint = hashlib.sha256(
                    np.asarray(coordinates, dtype=np.int32).tobytes()
                    + f"{int(level)}:{side}".encode()
                ).hexdigest()[:16]
                records.append(
                    {
                        "case_id": case_id,
                        "reviewer_id": "",
                        "cluster_id": next_id,
                        "cluster_key": cluster_key,
                        "cluster_fingerprint": fingerprint,
                        "level": int(level),
                        "side": side,
                        "component_id": int(component_id),
                        "voxel_count": int(np.count_nonzero(component)),
                        "slice_start_ras": int(np.min(z_values)),
                        "slice_stop_ras": int(np.max(z_values)),
                        "centroid_x_ras": round(float(centroid[0]), 3),
                        "centroid_y_ras": round(float(centroid[1]), 3),
                        "centroid_z_ras": round(float(centroid[2]), 3),
                        "median_attachment_ap_mm": round(median_ap, 4),
                        "minimum_cord_distance_mm": round(minimum_distance, 4),
                        "suggested_class": suggestion,
                        "expert_class": "",
                        "attachment_visible": "",
                        "reviewer_confidence": "",
                        "notes": "",
                        "updated_at": "",
                    }
                )
                next_id += 1
    return cluster_map, records


def load_case(
    anatomy_path: Path,
    rootlets_path: Path,
    cord_path: Path,
    *,
    case_id: str,
) -> AnnotationCase:
    anatomy_path = anatomy_path.expanduser().resolve()
    rootlets_path = rootlets_path.expanduser().resolve()
    cord_path = cord_path.expanduser().resolve()
    anatomy_image = nib.load(anatomy_path)
    rootlets_image = nib.load(rootlets_path)
    cord_image = nib.load(cord_path)
    if not _same_grid(anatomy_image, rootlets_image) or not _same_grid(
        rootlets_image, cord_image
    ):
        raise ValueError("Anatomy, RootletSeg, and cord masks must use one voxel grid.")

    anatomy_ras, anatomy_transform = _to_canonical(
        np.asanyarray(anatomy_image.dataobj), anatomy_image.affine
    )
    rootlets_ras, rootlet_transform = _to_canonical(
        np.asanyarray(rootlets_image.dataobj), rootlets_image.affine
    )
    cord_ras, cord_transform = _to_canonical(
        np.asanyarray(cord_image.dataobj), cord_image.affine
    )
    if not (
        np.array_equal(anatomy_transform, rootlet_transform)
        and np.array_equal(rootlet_transform, cord_transform)
    ):
        raise ValueError("Input orientation transforms differ.")
    canonical_affine = rootlets_image.affine @ nib.orientations.inv_ornt_aff(
        rootlet_transform, rootlets_image.shape
    )
    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(canonical_affine))
    cluster_map, records = extract_rootlet_clusters(
        rootlets_ras,
        cord_ras,
        spacing,
        affine=canonical_affine,
        case_id=case_id,
    )
    return AnnotationCase(
        case_id=case_id,
        anatomy_path=anatomy_path,
        rootlets_path=rootlets_path,
        cord_path=cord_path,
        anatomy_image=anatomy_image,
        rootlets_image=rootlets_image,
        anatomy_ras=np.asarray(anatomy_ras, dtype=np.float32),
        rootlets_ras=np.rint(rootlets_ras).astype(np.int16),
        cord_ras=np.asarray(cord_ras) > 0,
        cluster_map_ras=cluster_map,
        records=records,
    )


def merge_saved_annotations(
    records: list[dict[str, Any]], saved_rows: Iterable[dict[str, str]]
) -> list[dict[str, Any]]:
    saved = {row["cluster_key"]: row for row in saved_rows}
    if saved and set(saved) != {record["cluster_key"] for record in records}:
        raise ValueError(
            "Saved review clusters do not match this segmentation; use a new output directory."
        )
    for record in records:
        previous = saved.get(record["cluster_key"])
        if previous:
            if previous.get("cluster_fingerprint") != record["cluster_fingerprint"]:
                raise ValueError(
                    "Saved review geometry does not match this segmentation; "
                    "use a new output directory."
                )
            for field in (
                "expert_class",
                "reviewer_id",
                "attachment_visible",
                "reviewer_confidence",
                "notes",
                "updated_at",
            ):
                record[field] = previous.get(field, "")
    return records


def read_review_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_review_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def annotate_record(
    record: dict[str, Any],
    expert_class: str,
    *,
    reviewer_id: str,
    visible: str,
    confidence: str,
    notes: str,
) -> None:
    if expert_class not in EXPERT_CLASSES:
        raise ValueError(f"Invalid expert class: {expert_class!r}.")
    if not reviewer_id.strip():
        raise ValueError("reviewer_id must not be empty.")
    if visible not in {"yes", "no", "unclear"}:
        raise ValueError(f"Invalid visibility: {visible!r}.")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid confidence: {confidence!r}.")
    record.update(
        {
            "expert_class": expert_class,
            "reviewer_id": reviewer_id.strip(),
            "attachment_visible": visible,
            "reviewer_confidence": confidence,
            "notes": notes.strip(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def export_annotation_dataset(
    case: AnnotationCase,
    output_directory: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    masks = {
        label: np.zeros(case.rootlets_ras.shape, dtype=np.int16)
        for label in (*EXPERT_CLASSES, "unreviewed")
    }
    record_by_id = {int(record["cluster_id"]): record for record in records}
    for cluster_id, record in record_by_id.items():
        label = record["expert_class"] or "unreviewed"
        support = case.cluster_map_ras == cluster_id
        masks[label][support] = case.rootlets_ras[support]

    output_paths: dict[str, str] = {}
    for label, data in masks.items():
        native = _from_canonical(data, case.rootlets_image.affine)
        path = output_directory / (
            f"{case.case_id}_desc-{label}_label-rootlets_dseg.nii.gz"
        )
        _save_like(native, case.rootlets_image, path, np.int16)
        output_paths[label] = str(path.resolve())

    cluster_native = _from_canonical(
        case.cluster_map_ras, case.rootlets_image.affine
    )
    cluster_path = output_directory / (
        f"{case.case_id}_desc-clusters_label-rootlets_dseg.nii.gz"
    )
    _save_like(cluster_native, case.rootlets_image, cluster_path, np.int32)
    output_paths["cluster_map"] = str(cluster_path.resolve())

    counts = {
        label: sum((record["expert_class"] or "unreviewed") == label for record in records)
        for label in (*EXPERT_CLASSES, "unreviewed")
    }
    reviewer_ids = sorted(
        {record["reviewer_id"] for record in records if record["reviewer_id"]}
    )
    manifest = {
        "schema": "rootlet-dorsal-ventral-annotations-v1",
        "case_id": case.case_id,
        "reviewer_ids": reviewer_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "anatomy": str(case.anatomy_path),
            "rootlets": str(case.rootlets_path),
            "cord": str(case.cord_path),
        },
        "cluster_count": len(records),
        "class_counts": counts,
        "outputs": output_paths,
        "training_contract": (
            "Only dorsal/ventral expert classes are supervised targets; mixed, "
            "unclear, and unreviewed clusters must be excluded."
        ),
    }
    manifest_path = output_directory / f"{case.case_id}_annotations.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
