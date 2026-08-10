#!/usr/bin/env python3
"""Generate shared cropped spinal-cord masks for the V2-V4 comparison."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np


def _bounds(
    mask: np.ndarray,
    spacing: tuple[float, ...],
    padding_mm: float,
) -> tuple[slice, ...]:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        raise ValueError("Rootlet support is empty.")
    bounds: list[slice] = []
    for axis, values in enumerate(coordinates):
        padding = int(math.ceil(padding_mm / spacing[axis]))
        bounds.append(
            slice(
                max(0, int(values.min()) - padding),
                min(mask.shape[axis], int(values.max()) + padding + 1),
            )
        )
    return tuple(bounds)


def _crop_affine(affine: np.ndarray, bounds: tuple[slice, ...]) -> np.ndarray:
    translation = np.eye(4)
    translation[:3, 3] = [axis.start for axis in bounds]
    return affine @ translation


def prepare(
    dataset_directory: Path,
    output_directory: Path,
    timing_directory: Path,
    *,
    sct_command: Path,
    padding_mm: float,
    centerline: str,
) -> dict[str, object]:
    manifest = json.loads((dataset_directory / "fallback_manifest.json").read_text())
    output_directory.mkdir(parents=True, exist_ok=True)
    timing_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for case_id in manifest["split"]["test"]:
        started = time.perf_counter()
        anatomy = nib.load(dataset_directory / "imagesTs" / f"{case_id}_0000.nii.gz")
        rootlet_image = nib.load(
            dataset_directory / "imagesTs" / f"{case_id}_0001.nii.gz"
        )
        if anatomy.shape != rootlet_image.shape or not np.allclose(
            anatomy.affine, rootlet_image.affine, atol=1e-4
        ):
            raise ValueError(f"{case_id}: anatomy and rootlet grids differ.")
        rootlets = np.asanyarray(rootlet_image.dataobj) > 0
        spacing = tuple(float(value) for value in anatomy.header.get_zooms()[:3])
        bounds = _bounds(rootlets, spacing, padding_mm)
        crop = np.asanyarray(anatomy.dataobj)[bounds]
        crop_affine = _crop_affine(anatomy.affine, bounds)

        with tempfile.TemporaryDirectory(prefix="rootlet-dv-cord-") as temporary:
            directory = Path(temporary)
            crop_path = directory / "anatomy_crop.nii.gz"
            crop_segmentation_path = directory / "cord_crop.nii.gz"
            nib.save(
                nib.Nifti1Image(crop, crop_affine, header=anatomy.header.copy()),
                crop_path,
            )
            command = [
                str(sct_command),
                "-i",
                str(crop_path),
                "-c",
                "t1",
                "-centerline",
                centerline,
            ]
            if centerline == "cnn":
                command.extend(("-brain", "0"))
            command.extend(("-o", str(crop_segmentation_path)))
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            log_path = timing_directory / f"{case_id}_cord_crop.log"
            log_path.write_text(completed.stdout + completed.stderr)
            if completed.returncode != 0 or not crop_segmentation_path.is_file():
                raise RuntimeError(
                    f"{case_id}: SCT cord segmentation failed; see {log_path}."
                )
            cropped_cord_image = nib.load(crop_segmentation_path)
            cropped_cord = np.asanyarray(cropped_cord_image.dataobj) > 0
            if cropped_cord.shape != crop.shape:
                raise ValueError(f"{case_id}: SCT changed the cropped grid shape.")

        full_cord = np.zeros(anatomy.shape, dtype=np.uint8)
        full_cord[bounds] = cropped_cord.astype(np.uint8)
        if not np.any(full_cord):
            raise ValueError(f"{case_id}: SCT returned an empty cord mask.")
        output_path = output_directory / f"{case_id}_label-SC_seg.nii.gz"
        header = anatomy.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(full_cord, anatomy.affine, header), output_path)
        duration = time.perf_counter() - started
        (timing_directory / f"{case_id}_cord_crop.time").write_text(
            f"Elapsed (wall clock) time (h:mm:ss or m:ss): {duration:.6f}\n"
        )
        records.append(
            {
                "case": case_id,
                "crop_shape": list(crop.shape),
                "cord_voxels": int(np.count_nonzero(full_cord)),
                "seconds": duration,
            }
        )

    result = {
        "schema": "rootlet-dv-shared-cord-preparation-v1",
        "method": f"legacy SCT deepseg_sc t1, {centerline} centerline, 2-D kernel",
        "padding_mm": padding_mm,
        "cases": records,
    }
    (timing_directory / "cord_crop_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--timing-directory", required=True, type=Path)
    parser.add_argument("--sct-command", required=True, type=Path)
    parser.add_argument("--padding-mm", type=float, default=20.0)
    parser.add_argument("--centerline", choices=("cnn", "svm"), default="cnn")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = prepare(
        args.dataset_directory,
        args.output_directory,
        args.timing_directory,
        sct_command=args.sct_command,
        padding_mm=args.padding_mm,
        centerline=args.centerline,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
