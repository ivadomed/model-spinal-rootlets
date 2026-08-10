#!/usr/bin/env python3
"""Stage MRI and RootletSeg pairs as strict RPI inputs for hybrid V5 inference."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.cluster_mean_v5 import (
    RPI_ORIENTATION,
    split_cluster_mean_v5,
)


REQUIRED_COLUMNS = {"case_id", "dataset", "image", "rootlets"}
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _load_integer_labels(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    values = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Rootlet labels contain non-finite values: {path}")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"Rootlet labels must be integer-valued: {path}")
    if np.any(rounded < 0) or np.any(rounded > np.iinfo(np.int16).max):
        raise ValueError(f"Rootlet labels are outside the int16 range: {path}")
    return image, rounded.astype(np.int16)


def _to_rpi(data: np.ndarray, image: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    orientation = nib.orientations.io_orientation(image.affine)
    transform = nib.orientations.ornt_transform(orientation, RPI_ORIENTATION)
    rpi = nib.orientations.apply_orientation(data, transform)
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape)
    return rpi, affine


def _save(data: np.ndarray, affine: np.ndarray, path: Path, dtype: np.dtype) -> None:
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    output = nib.Nifti1Image(data.astype(dtype), affine, header=header)
    output.set_qform(affine, 1)
    output.set_sform(affine, 1)
    nib.save(output, path)


def stage(manifest: Path, output_directory: Path) -> dict[str, Any]:
    """Validate and stage all manifest rows, returning a machine-readable audit."""

    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest has no cases.")

    output_directory.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["case_id"]
        if not CASE_ID.fullmatch(case_id):
            raise ValueError(f"Invalid nnU-Net case ID: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"Duplicate case ID: {case_id}")
        seen.add(case_id)
        image_path = _resolve(row["image"], manifest)
        rootlet_path = _resolve(row["rootlets"], manifest)
        anatomy_image = nib.load(image_path)
        anatomy = np.asanyarray(anatomy_image.dataobj)
        rootlet_image, rootlets = _load_integer_labels(rootlet_path)
        if anatomy_image.shape != rootlet_image.shape or not np.allclose(
            anatomy_image.affine, rootlet_image.affine, atol=1e-4
        ):
            raise ValueError(f"MRI and RootletSeg grids differ for {case_id}.")
        if not np.any(rootlets > 0):
            raise ValueError(f"RootletSeg support is empty for {case_id}.")

        anatomy_rpi, anatomy_affine = _to_rpi(anatomy, anatomy_image)
        rootlets_rpi, rootlet_affine = _to_rpi(rootlets, rootlet_image)
        if anatomy_rpi.shape != rootlets_rpi.shape or not np.allclose(
            anatomy_affine, rootlet_affine, atol=1e-4
        ):
            raise RuntimeError(f"RPI conversion changed paired grids for {case_id}.")
        if tuple(nib.aff2axcodes(rootlet_affine)) != ("R", "P", "I"):
            raise RuntimeError(f"RPI conversion failed for {case_id}.")

        image_output = output_directory / f"{case_id}_0000.nii.gz"
        rootlet_output = output_directory / f"{case_id}_0001.nii.gz"
        _save(anatomy_rpi, anatomy_affine, image_output, np.float32)
        _save(rootlets_rpi, rootlet_affine, rootlet_output, np.int16)

        spacing_y_mm = float(nib.affines.voxel_sizes(rootlet_affine)[1])
        v5 = split_cluster_mean_v5(rootlets_rpi, spacing_y_mm)
        total_levels = len(v5.qc["levels"])
        deterministic_levels = int(v5.qc["deterministic_levels"])
        cases.append(
            {
                "case_id": case_id,
                "dataset": row["dataset"],
                "source_image": str(image_path.resolve()),
                "source_rootlets": str(rootlet_path.resolve()),
                "rpi_image": str(image_output.resolve()),
                "rpi_rootlets": str(rootlet_output.resolve()),
                "rootlet_voxels": int(np.count_nonzero(rootlets_rpi)),
                "levels": total_levels,
                "deterministic_levels": deterministic_levels,
                "fallback_levels": total_levels - deterministic_levels,
                "v5_level_coverage": deterministic_levels / total_levels,
            }
        )

    audit = {
        "schema": "rootlet-dv-hybrid-v5-inference-stage-v1",
        "orientation": "RPI",
        "channels": {"0": "MRI", "1": "level-labelled RootletSeg support"},
        "case_count": len(cases),
        "datasets": sorted(set(case["dataset"] for case in cases)),
        "cases": cases,
    }
    (output_directory / "stage_manifest.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    return audit


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = stage(args.manifest, args.output_directory)
    print(json.dumps({"case_count": result["case_count"], "datasets": result["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
