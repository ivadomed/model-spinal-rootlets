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
    _interpolated_cord_centres,
    _local_surface_coordinates,
    _save_like,
    _validate_inputs,
)


EXPERT_CLASSES = ("dorsal", "ventral", "mixed", "unclear")
MERGED_DV_CLASS = "split_dv"
RPI_ORIENTATION = nib.orientations.axcodes2ornt(("R", "P", "I"))
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
    "slice_start_rpi",
    "slice_stop_rpi",
    "centroid_x_rpi",
    "centroid_y_rpi",
    "centroid_z_rpi",
    "median_attachment_ap_mm",
    "minimum_cord_distance_mm",
    "suggested_class",
    "expert_class",
    "split_method",
    "split_cut_rpi",
    "attachment_visible",
    "reviewer_confidence",
    "notes",
    "updated_at",
)


@dataclass
class AnnotationCase:
    """Loaded images, RPI arrays, and deterministic cluster records."""

    case_id: str
    anatomy_path: Path
    rootlets_path: Path
    cord_path: Path | None
    anatomy_image: nib.Nifti1Image
    rootlets_image: nib.Nifti1Image
    anatomy_rpi: np.ndarray
    rootlets_rpi: np.ndarray
    cord_rpi: np.ndarray
    cluster_map_rpi: np.ndarray
    anatomy_rpi_image: nib.Nifti1Image
    rootlets_rpi_image: nib.Nifti1Image
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class DiscoveredCase:
    """An anatomy/rootlet-label pair, with an optional spinal-cord mask."""

    case_id: str
    anatomy_path: Path
    rootlets_path: Path
    cord_path: Path | None


def _to_rpi(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a volume to RPI voxel order without changing its world space."""

    original_orientation = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(original_orientation, RPI_ORIENTATION)
    return nib.orientations.apply_orientation(data, transform), transform


def _rpi_image(
    data: np.ndarray, source: nib.Nifti1Image, transform: np.ndarray
) -> nib.Nifti1Image:
    """Build an RPI NIfTI with the original image's physical coordinates."""

    affine = source.affine @ nib.orientations.inv_ornt_aff(transform, source.shape)
    header = source.header.copy()
    header.set_data_dtype(data.dtype)
    image = nib.Nifti1Image(data, affine, header)
    image.set_qform(affine, int(source.header["qform_code"]))
    image.set_sform(affine, int(source.header["sform_code"]))
    return image


def split_merged_component_ap(
    cluster_map_rpi: np.ndarray, cluster_id: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bisect one manually flagged merged rootlet at its AP two-mode midpoint.

    In RPI voxel order, lower Y is anterior/ventral and higher Y is
    posterior/dorsal. This only provides the geometry after an expert explicitly
    identifies a component as a dorsal-plus-ventral merge; it is never applied
    automatically.
    """

    component = cluster_map_rpi == int(cluster_id)
    coordinates = np.argwhere(component)
    if len(coordinates) < 2:
        raise ValueError("A merged rootlet needs at least two voxels to split.")
    ap = coordinates[:, 1].astype(float)
    lower, upper = np.percentile(ap, (25, 75))
    if lower == upper:
        raise ValueError("This rootlet has no AP extent to split.")
    for _ in range(20):
        midpoint = (lower + upper) / 2.0
        lower_values = ap[ap <= midpoint]
        upper_values = ap[ap > midpoint]
        if not len(lower_values) or not len(upper_values):
            raise ValueError("This rootlet has no separable AP branches.")
        new_lower = float(np.mean(lower_values))
        new_upper = float(np.mean(upper_values))
        if np.isclose(new_lower, lower) and np.isclose(new_upper, upper):
            break
        lower, upper = new_lower, new_upper
    split_cut = (lower + upper) / 2.0
    ventral = np.zeros_like(component, dtype=bool)
    dorsal = np.zeros_like(component, dtype=bool)
    ventral_coordinates = coordinates[ap <= split_cut]
    dorsal_coordinates = coordinates[ap > split_cut]
    ventral[tuple(ventral_coordinates.T)] = True
    dorsal[tuple(dorsal_coordinates.T)] = True
    if not np.any(ventral) or not np.any(dorsal):
        raise ValueError("This rootlet has no separable AP branches.")
    return dorsal, ventral, float(split_cut)


def _nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def discover_cases(
    search_root: Path, *, max_files: int = 5000
) -> list[DiscoveredCase]:
    """Find complete BIDS-like input triplets below ``search_root``."""
    search_root = search_root.expanduser().resolve()
    if not search_root.exists():
        raise ValueError(f"Search directory does not exist: {search_root}")
    if not search_root.is_dir():
        raise ValueError(f"Search directory must be a folder: {search_root}")

    nifti_files: list[Path] = []
    for path in search_root.rglob("*"):
        if path.is_file() and (
            path.name.lower().endswith(".nii")
            or path.name.lower().endswith(".nii.gz")
        ):
            nifti_files.append(path.resolve())
            if len(nifti_files) > max_files:
                raise ValueError(
                    f"Search found more than {max_files} NIfTI files. "
                    "Choose a smaller dataset folder."
                )

    by_name: dict[str, list[Path]] = {}
    for path in sorted(nifti_files):
        by_name.setdefault(path.name, []).append(path)

    discovered: list[DiscoveredCase] = []
    for rootlets_path in nifti_files:
        stem = _nifti_stem(rootlets_path)
        marker = "_label-rootlets_dseg"
        if not stem.endswith(marker):
            continue
        case_id = stem[: -len(marker)]
        extension = (
            ".nii.gz" if rootlets_path.name.lower().endswith(".nii.gz") else ".nii"
        )
        cord_candidates = by_name.get(f"{case_id}_label-SC_seg{extension}", [])
        anatomy_candidates: list[Path] = []
        for suffix in ("T2w", "T1w", "T2star", "T2starw"):
            anatomy_candidates.extend(
                by_name.get(f"{case_id}_{suffix}{extension}", [])
            )
        if not cord_candidates or not anatomy_candidates:
            continue
        cord_path = min(
            cord_candidates,
            key=lambda path: (path.parent != rootlets_path.parent, str(path)),
        )
        anatomy_path = min(
            anatomy_candidates,
            key=lambda path: (path.parent != rootlets_path.parent, str(path)),
        )
        discovered.append(
            DiscoveredCase(
                case_id=case_id,
                anatomy_path=anatomy_path,
                rootlets_path=rootlets_path,
                cord_path=cord_path,
            )
        )
    return sorted(discovered, key=lambda case: (case.case_id, str(case.rootlets_path)))


def discover_reference_label_cases(
    search_root: Path, *, max_files: int = 5000, include_described: bool = False
) -> list[DiscoveredCase]:
    """Find MRI/rootlet-reference pairs without requiring a spinal-cord mask.

    The rootlet training datasets store their manual label in a derivatives
    folder and the corresponding MRI elsewhere in the BIDS tree.  The case
    stem immediately before ``_label-rootlets_dseg`` is the shared filename.
    By default, rater/STAPLE ``desc-`` variants are excluded so one MRI cannot
    enter the annotation queue several times with competing label versions.
    """

    search_root = search_root.expanduser().resolve()
    if not search_root.exists():
        raise ValueError(f"Search directory does not exist: {search_root}")
    if not search_root.is_dir():
        raise ValueError(f"Search directory must be a folder: {search_root}")

    marker = "_label-rootlets_dseg"
    rootlet_cases: list[tuple[str, Path, str]] = []
    for rootlets_path in search_root.rglob("*_label-rootlets_dseg.nii*"):
        rootlets_path = rootlets_path.resolve()
        stem = _nifti_stem(rootlets_path)
        case_id = stem[: -len(marker)]
        if not include_described and "_desc-" in case_id:
            continue
        extension = (
            ".nii.gz" if rootlets_path.name.lower().endswith(".nii.gz") else ".nii"
        )
        rootlet_cases.append((case_id, rootlets_path, f"{case_id}{extension}"))
        if len(rootlet_cases) > max_files:
            raise ValueError(
                f"Search found more than {max_files} rootlet-label files. "
                "Choose a smaller dataset folder."
            )

    target_names = {image_name for _case_id, _rootlets_path, image_name in rootlet_cases}
    anatomy_by_name: dict[str, list[Path]] = {name: [] for name in target_names}
    for path in search_root.rglob("*"):
        if path.is_file() and path.name in anatomy_by_name:
            anatomy_by_name[path.name].append(path.resolve())

    discovered: list[DiscoveredCase] = []
    for case_id, rootlets_path, anatomy_name in rootlet_cases:
        anatomy_candidates = anatomy_by_name[anatomy_name]
        if not anatomy_candidates:
            continue
        anatomy_path = min(
            anatomy_candidates,
            key=lambda path: (path.parent != rootlets_path.parent, str(path)),
        )
        discovered.append(
            DiscoveredCase(
                case_id=case_id,
                anatomy_path=anatomy_path,
                rootlets_path=rootlets_path,
                cord_path=None,
            )
        )
    return sorted(discovered, key=lambda case: (case.case_id, str(case.rootlets_path)))


def discover_nnunet_label_cases(dataset_root: Path) -> list[DiscoveredCase]:
    """Find standard nnU-Net ``imagesTr``/``labelsTr`` reference-label pairs."""

    dataset_root = dataset_root.expanduser().resolve()
    images_directory = dataset_root / "imagesTr"
    labels_directory = dataset_root / "labelsTr"
    if not images_directory.is_dir() or not labels_directory.is_dir():
        raise ValueError(
            "nnU-Net label discovery needs both imagesTr and labelsTr directories."
        )
    discovered: list[DiscoveredCase] = []
    for rootlets_path in sorted(labels_directory.glob("*.nii*")):
        case_id = _nifti_stem(rootlets_path)
        anatomy_candidates = [
            images_directory / f"{case_id}_0000.nii.gz",
            images_directory / f"{case_id}_0000.nii",
        ]
        anatomy_path = next((path for path in anatomy_candidates if path.is_file()), None)
        if anatomy_path is None:
            continue
        discovered.append(
            DiscoveredCase(
                case_id=case_id,
                anatomy_path=anatomy_path.resolve(),
                rootlets_path=rootlets_path.resolve(),
                cord_path=None,
            )
        )
    return discovered


def _load_nifti(path: Path, field: str) -> tuple[Path, nib.Nifti1Image]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{field} does not exist: {resolved}")
    if resolved.is_dir():
        raise ValueError(f"{field} must be a NIfTI file, not a folder: {resolved}")
    if not (
        resolved.name.lower().endswith(".nii")
        or resolved.name.lower().endswith(".nii.gz")
    ):
        raise ValueError(f"{field} must end in .nii or .nii.gz: {resolved}")
    try:
        image = nib.load(resolved)
    except Exception as error:
        raise ValueError(
            f"Could not load {field} as NIfTI: {resolved} ({error})"
        ) from error
    return resolved, image


def _same_grid(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return first.shape == second.shape and np.allclose(
        first.affine, second.affine, atol=1e-4
    )


def _validate_rootlet_labels(rootlets: np.ndarray) -> np.ndarray:
    """Validate a level-labelled rootlet mask when no cord mask is available."""

    if rootlets.ndim != 3:
        raise ValueError("Rootlet labels must be a 3-D volume.")
    if not np.all(np.isfinite(rootlets)):
        raise ValueError("Rootlet labels must not contain NaN or infinite values.")
    rounded = np.rint(rootlets)
    if not np.allclose(rootlets, rounded, atol=1e-4):
        raise ValueError("Rootlet labels must contain integer values.")
    labels = rounded.astype(np.int32)
    if np.any(labels < 0):
        raise ValueError("Rootlet labels must be non-negative.")
    if not np.any(labels > 0):
        raise ValueError("Rootlet labels are empty.")
    return labels


def extract_rootlet_clusters(
    rootlets: np.ndarray,
    cord: np.ndarray | None,
    spacing: tuple[float, float, float],
    *,
    affine: np.ndarray | None = None,
    case_id: str = "",
    attachment_distance_mm: float = 4.0,
    attachment_band_mm: float = 0.8,
    ap_margin_mm: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract full connected rootlet branches, separated by level and side."""
    if cord is None:
        labels = _validate_rootlet_labels(rootlets)
        cluster_map = np.zeros(labels.shape, dtype=np.int32)
        records: list[dict[str, Any]] = []
        structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
        next_id = 1
        for level in np.unique(labels[labels > 0]):
            components, _ = ndimage.label(labels == level, structure=structure)
            for component_id, component_slice in enumerate(
                ndimage.find_objects(components), start=1
            ):
                if component_slice is None:
                    continue
                component = components[component_slice] == component_id
                map_view = cluster_map[component_slice]
                map_view[component] = next_id
                coordinates = np.argwhere(component)
                coordinates += np.array([axis.start for axis in component_slice])
                centroid = np.mean(coordinates, axis=0)
                z_values = coordinates[:, 2]
                cluster_key = f"L{int(level):02d}-unassigned-C{component_id:02d}"
                fingerprint = hashlib.sha256(
                    np.asarray(coordinates, dtype=np.int32).tobytes()
                    + f"{int(level)}:unassigned".encode()
                ).hexdigest()[:16]
                records.append(
                    {
                        "case_id": case_id,
                        "reviewer_id": "",
                        "cluster_id": next_id,
                        "cluster_key": cluster_key,
                        "cluster_fingerprint": fingerprint,
                        "level": int(level),
                        "side": "unassigned",
                        "component_id": int(component_id),
                        "voxel_count": int(np.count_nonzero(component)),
                        "slice_start_rpi": int(np.min(z_values)),
                        "slice_stop_rpi": int(np.max(z_values)),
                        "centroid_x_rpi": round(float(centroid[0]), 3),
                        "centroid_y_rpi": round(float(centroid[1]), 3),
                        "centroid_z_rpi": round(float(centroid[2]), 3),
                        "median_attachment_ap_mm": "",
                        "minimum_cord_distance_mm": "",
                        "suggested_class": "",
                        "expert_class": "",
                        "split_method": "",
                        "split_cut_rpi": "",
                        "attachment_visible": "",
                        "reviewer_confidence": "",
                        "notes": "",
                        "updated_at": "",
                    }
                )
                next_id += 1
        return cluster_map, records

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
                        "slice_start_rpi": int(np.min(z_values)),
                        "slice_stop_rpi": int(np.max(z_values)),
                        "centroid_x_rpi": round(float(centroid[0]), 3),
                        "centroid_y_rpi": round(float(centroid[1]), 3),
                        "centroid_z_rpi": round(float(centroid[2]), 3),
                        "median_attachment_ap_mm": round(median_ap, 4),
                        "minimum_cord_distance_mm": round(minimum_distance, 4),
                        "suggested_class": suggestion,
                        "expert_class": "",
                        "split_method": "",
                        "split_cut_rpi": "",
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
    cord_path: Path | None,
    *,
    case_id: str,
) -> AnnotationCase:
    anatomy_path, anatomy_image = _load_nifti(anatomy_path, "Anatomical image")
    rootlets_path, rootlets_image = _load_nifti(rootlets_path, "RootletSeg image")
    if not _same_grid(anatomy_image, rootlets_image):
        raise ValueError("Anatomy and rootlet labels must use one voxel grid.")
    cord_image: nib.Nifti1Image | None = None
    resolved_cord_path: Path | None = None
    if cord_path is not None:
        resolved_cord_path, cord_image = _load_nifti(cord_path, "Spinal cord mask")
        if not _same_grid(rootlets_image, cord_image):
            raise ValueError("Rootlet labels and cord mask must use one voxel grid.")

    anatomy_rpi, anatomy_transform = _to_rpi(
        np.asanyarray(anatomy_image.dataobj), anatomy_image.affine
    )
    rootlets_rpi, rootlet_transform = _to_rpi(
        np.asanyarray(rootlets_image.dataobj), rootlets_image.affine
    )
    if not np.array_equal(anatomy_transform, rootlet_transform):
        raise ValueError("Input orientation transforms differ.")
    if cord_image is not None:
        cord_rpi, cord_transform = _to_rpi(
            np.asanyarray(cord_image.dataobj), cord_image.affine
        )
        if not np.array_equal(rootlet_transform, cord_transform):
            raise ValueError("Input orientation transforms differ.")
    else:
        cord_rpi = np.zeros_like(rootlets_rpi, dtype=bool)
    rpi_affine = rootlets_image.affine @ nib.orientations.inv_ornt_aff(
        rootlet_transform, rootlets_image.shape
    )
    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(rpi_affine))
    cluster_map, records = extract_rootlet_clusters(
        rootlets_rpi,
        cord_rpi if cord_image is not None else None,
        spacing,
        affine=rpi_affine,
        case_id=case_id,
    )
    return AnnotationCase(
        case_id=case_id,
        anatomy_path=anatomy_path,
        rootlets_path=rootlets_path,
        cord_path=resolved_cord_path,
        anatomy_image=anatomy_image,
        rootlets_image=rootlets_image,
        anatomy_rpi=np.asarray(anatomy_rpi, dtype=np.float32),
        rootlets_rpi=np.rint(rootlets_rpi).astype(np.int16),
        cord_rpi=np.asarray(cord_rpi) > 0,
        cluster_map_rpi=cluster_map,
        anatomy_rpi_image=_rpi_image(anatomy_rpi, anatomy_image, anatomy_transform),
        rootlets_rpi_image=_rpi_image(
            np.rint(rootlets_rpi).astype(np.int16), rootlets_image, rootlet_transform
        ),
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
                "split_method",
                "split_cut_rpi",
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
            "split_method": "",
            "split_cut_rpi": "",
            "reviewer_id": reviewer_id.strip(),
            "attachment_visible": visible,
            "reviewer_confidence": confidence,
            "notes": notes.strip(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def annotate_merged_dv(
    record: dict[str, Any],
    *,
    split_cut_rpi: float,
    reviewer_id: str,
) -> None:
    """Record an expert-approved AP split of one merged D/V component."""

    if not reviewer_id.strip():
        raise ValueError("reviewer_id must not be empty.")
    record.update(
        {
            "expert_class": MERGED_DV_CLASS,
            "split_method": "expert-approved-ap-two-means",
            "split_cut_rpi": round(float(split_cut_rpi), 4),
            "reviewer_id": reviewer_id.strip(),
            "attachment_visible": "yes",
            "reviewer_confidence": "medium",
            "notes": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _saved_merged_dv_masks(
    case: AnnotationCase, record: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the expert-approved AP split from the saved review record."""

    try:
        split_cut = float(record["split_cut_rpi"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Merged D/V record is missing its AP split location.") from error
    component = case.cluster_map_rpi == int(record["cluster_id"])
    coordinates = np.argwhere(component)
    ventral = np.zeros_like(component, dtype=bool)
    dorsal = np.zeros_like(component, dtype=bool)
    ventral_coordinates = coordinates[coordinates[:, 1] <= split_cut]
    dorsal_coordinates = coordinates[coordinates[:, 1] > split_cut]
    ventral[tuple(ventral_coordinates.T)] = True
    dorsal[tuple(dorsal_coordinates.T)] = True
    if not np.any(ventral) or not np.any(dorsal):
        raise ValueError("Merged D/V split no longer divides the rootlet component.")
    return dorsal, ventral


def export_annotation_dataset(
    case: AnnotationCase,
    output_directory: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    masks = {
        label: np.zeros(case.rootlets_rpi.shape, dtype=np.int16)
        for label in (*EXPERT_CLASSES, "unreviewed")
    }
    record_by_id = {int(record["cluster_id"]): record for record in records}
    for cluster_id, record in record_by_id.items():
        label = record["expert_class"] or "unreviewed"
        support = case.cluster_map_rpi == cluster_id
        if label == MERGED_DV_CLASS:
            dorsal, ventral = _saved_merged_dv_masks(case, record)
            masks["dorsal"][dorsal] = case.rootlets_rpi[dorsal]
            masks["ventral"][ventral] = case.rootlets_rpi[ventral]
            continue
        masks[label][support] = case.rootlets_rpi[support]

    output_paths: dict[str, str] = {}
    for label, data in masks.items():
        path = output_directory / (
            f"{case.case_id}_desc-{label}_label-rootlets_dseg.nii.gz"
        )
        _save_like(data, case.rootlets_rpi_image, path, np.int16)
        output_paths[label] = str(path.resolve())

    cluster_path = output_directory / (
        f"{case.case_id}_desc-clusters_label-rootlets_dseg.nii.gz"
    )
    _save_like(case.cluster_map_rpi, case.rootlets_rpi_image, cluster_path, np.int32)
    output_paths["cluster_map"] = str(cluster_path.resolve())

    counts = {
        label: sum((record["expert_class"] or "unreviewed") == label for record in records)
        for label in (*EXPERT_CLASSES, "unreviewed")
    }
    counts["merged_dv"] = sum(
        record["expert_class"] == MERGED_DV_CLASS for record in records
    )
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
            "cord": str(case.cord_path) if case.cord_path else None,
        },
        "cluster_count": len(records),
        "class_counts": counts,
        "orientation": "RPI",
        "outputs": output_paths,
        "training_contract": (
            "Only dorsal/ventral expert classes are supervised targets; mixed, "
            "unclear, and unreviewed clusters must be excluded."
        ),
    }
    manifest_path = output_directory / f"{case.case_id}_annotations.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _dorsal_ventral_target(
    case: AnnotationCase, records: list[dict[str, Any]]
) -> np.ndarray:
    """Create the two-class target only when every rootlet cluster is reviewed."""

    target = np.zeros(case.rootlets_rpi.shape, dtype=np.uint8)
    expected_ids = {int(record["cluster_id"]) for record in records}
    actual_ids = {int(value) for value in np.unique(case.cluster_map_rpi) if value > 0}
    if actual_ids != expected_ids:
        raise ValueError("Cluster records do not match the current rootlet label map.")
    for record in records:
        label = record["expert_class"]
        if label == MERGED_DV_CLASS:
            dorsal, ventral = _saved_merged_dv_masks(case, record)
            target[dorsal] = 1
            target[ventral] = 2
            continue
        if label not in {"dorsal", "ventral"}:
            raise ValueError(
                "nnU-Net export requires every cluster to be labelled dorsal or ventral."
            )
        target[case.cluster_map_rpi == int(record["cluster_id"])] = (
            1 if label == "dorsal" else 2
        )
    if not np.array_equal(target > 0, case.rootlets_rpi > 0):
        raise ValueError("D/V target must cover the complete fixed rootlet support.")
    return target


def export_nnunet_case(
    case: AnnotationCase,
    output_directory: Path,
    records: list[dict[str, Any]],
) -> dict[str, str]:
    """Write one complete D/V case as two nnU-Net input channels and one target.

    Channel 0 is the anatomical MRI. Channel 1 is the level-labelled rootlet
    support used while reviewing. At inference it must be replaced by the
    first RootletSeg model's output on the same grid, never by a D/V target.
    Expert-approved merged components are stored as AP-split D/V targets.
    """

    case_id = case.case_id
    images_directory = output_directory / "imagesTr"
    labels_directory = output_directory / "labelsTr"
    images_directory.mkdir(parents=True, exist_ok=True)
    labels_directory.mkdir(parents=True, exist_ok=True)

    anatomy_path = images_directory / f"{case_id}_0000.nii.gz"
    rootlets_path = images_directory / f"{case_id}_0001.nii.gz"
    target_path = labels_directory / f"{case_id}.nii.gz"
    _save_like(case.anatomy_rpi, case.anatomy_rpi_image, anatomy_path, np.float32)
    # The source MRI and label affines can differ by harmless floating-point
    # noise even after the same RPI reorientation. nnU-Net requires channel and
    # target geometry metadata to match exactly, so write every array on the
    # MRI reference grid already validated by ``load_case``.
    _save_like(case.rootlets_rpi, case.anatomy_rpi_image, rootlets_path, np.float32)
    _save_like(
        _dorsal_ventral_target(case, records),
        case.anatomy_rpi_image,
        target_path,
        np.uint8,
    )
    return {
        "case_id": case_id,
        "anatomy": str(anatomy_path.resolve()),
        "rootlet_input": str(rootlets_path.resolve()),
        "target": str(target_path.resolve()),
    }


def write_nnunet_dataset_json(output_directory: Path) -> Path:
    """Refresh the nnU-Net dataset description after adding complete cases."""

    images_directory = output_directory / "imagesTr"
    case_count = len(list(images_directory.glob("*_0000.nii.gz"))) if images_directory.exists() else 0
    if not case_count:
        raise ValueError("Cannot create dataset.json without completed nnU-Net cases.")
    payload = {
        "channel_names": {
            "0": "MRI",
            "1": "rootlet-level-label-or-RootletSeg-prediction",
        },
        "labels": {"background": 0, "dorsal": 1, "ventral": 2},
        "numTraining": case_count,
        "file_ending": ".nii.gz",
        "description": (
            "Expert D/V labels on fixed rootlet support. Replace channel 1 with "
            "the first RootletSeg model output at inference."
        ),
    }
    dataset_json = output_directory / "dataset.json"
    temporary = dataset_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(dataset_json)
    return dataset_json
