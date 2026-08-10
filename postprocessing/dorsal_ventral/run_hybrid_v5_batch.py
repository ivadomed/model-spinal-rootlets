#!/usr/bin/env python3
"""Combine V5 with nnU-Net fallback predictions for a staged RPI cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.hybrid_v5 import (
    _load_probabilities,
    combine_v5_with_fallback,
)


def _save_like(data: np.ndarray, reference: nib.Nifti1Image, path: Path, dtype: np.dtype) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(data.astype(dtype), reference.affine, header=header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, path)


def run_batch(
    stage_directory: Path,
    prediction_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    stage = json.loads((stage_directory / "stage_manifest.json").read_text())
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for case in stage["cases"]:
        case_id = case["case_id"]
        rootlet_path = stage_directory / f"{case_id}_0001.nii.gz"
        prediction_path = prediction_directory / f"{case_id}.nii.gz"
        probability_path = prediction_directory / f"{case_id}.npz"
        rootlet_image = nib.load(rootlet_path)
        prediction_image = nib.load(prediction_path)
        if rootlet_image.shape != prediction_image.shape or not np.allclose(
            rootlet_image.affine, prediction_image.affine, atol=1e-4
        ):
            raise ValueError(f"Fallback prediction grid differs for {case_id}.")
        if tuple(nib.aff2axcodes(rootlet_image.affine)) != ("R", "P", "I"):
            raise ValueError(f"Staged case is not RPI: {case_id}.")
        rootlets = np.rint(np.asanyarray(rootlet_image.dataobj)).astype(np.int16)
        prediction = np.rint(np.asanyarray(prediction_image.dataobj)).astype(np.uint8)
        probabilities = (
            _load_probabilities(probability_path, rootlet_image.shape)
            if probability_path.is_file()
            else None
        )
        spacing_y_mm = float(nib.affines.voxel_sizes(rootlet_image.affine)[1])
        result = combine_v5_with_fallback(
            rootlets,
            prediction,
            spacing_y_mm,
            fallback_probabilities_rpi=probabilities,
        )

        dorsal_path = output_directory / f"{case_id}_desc-dorsal_label-rootlets_dseg.nii.gz"
        ventral_path = output_directory / f"{case_id}_desc-ventral_label-rootlets_dseg.nii.gz"
        class_path = output_directory / f"{case_id}_desc-hybridV5_dseg.nii.gz"
        qc_path = output_directory / f"{case_id}_desc-hybridV5_qc.json"
        _save_like(result.dorsal, rootlet_image, dorsal_path, np.int16)
        _save_like(result.ventral, rootlet_image, ventral_path, np.int16)
        _save_like(result.class_map, rootlet_image, class_path, np.uint8)
        qc_path.write_text(json.dumps(result.qc, indent=2, allow_nan=False) + "\n")
        support_voxels = int(np.count_nonzero(rootlets))
        fallback_voxels = int(result.qc["v5"]["fallback_voxels"])
        records.append(
            {
                **case,
                "dorsal": str(dorsal_path.resolve()),
                "ventral": str(ventral_path.resolve()),
                "class_map": str(class_path.resolve()),
                "qc": str(qc_path.resolve()),
                "support_voxels": support_voxels,
                "fallback_voxels": fallback_voxels,
                "fallback_voxel_fraction": fallback_voxels / support_voxels,
                "dorsal_fraction": int(np.count_nonzero(result.dorsal)) / support_voxels,
                "support_preserved": bool(result.qc["support_preserved"]),
            }
        )

    manifest_path = output_directory / "manifest.csv"
    with manifest_path.open("w", newline="") as stream:
        fields = (
            "case_id",
            "dataset",
            "rpi_image",
            "rpi_rootlets",
            "dorsal",
            "ventral",
            "class_map",
            "qc",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema": "rootlet-dv-hybrid-v5-batch-v1",
        "orientation": "RPI",
        "case_count": len(records),
        "dataset_count": len(set(record["dataset"] for record in records)),
        "exact_support_partitions": sum(record["support_preserved"] for record in records),
        "cases": records,
    }
    (output_directory / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-directory", required=True, type=Path)
    parser.add_argument("--prediction-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    summary = run_batch(
        args.stage_directory, args.prediction_directory, args.output_directory
    )
    print(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "exact_support_partitions": summary["exact_support_partitions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
