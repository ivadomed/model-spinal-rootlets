#!/usr/bin/env python3
"""Combine deterministic V5 decisions with a voxelwise fallback prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.cluster_mean_v5 import (
    RPI_ORIENTATION,
    split_cluster_mean_v5,
)


@dataclass(frozen=True)
class HybridV5Result:
    dorsal: np.ndarray
    ventral: np.ndarray
    class_map: np.ndarray
    qc: dict[str, Any]


def _fallback_classes(
    segmentation: np.ndarray,
    fallback_support: np.ndarray,
    probabilities: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    classes = np.rint(segmentation).astype(np.uint8)
    invalid = fallback_support & ~np.isin(classes, (1, 2))
    recovered = int(np.count_nonzero(invalid))
    if np.any(invalid):
        if probabilities is None:
            raise ValueError(
                "Fallback segmentation contains background on routed rootlet voxels; "
                "provide D/V probabilities so support can be preserved."
            )
        if probabilities.shape[0] < 3 or probabilities.shape[1:] != classes.shape:
            raise ValueError(
                "Fallback probabilities must have shape (classes, X, Y, Z) with "
                "dorsal=1 and ventral=2."
            )
        dv_choice = 1 + np.argmax(probabilities[1:3], axis=0).astype(np.uint8)
        classes[invalid] = dv_choice[invalid]
    if np.any(fallback_support & ~np.isin(classes, (1, 2))):
        raise RuntimeError("Internal error: fallback left routed rootlet voxels unlabelled.")
    return classes, recovered


def combine_v5_with_fallback(
    rootlets_rpi: np.ndarray,
    fallback_segmentation_rpi: np.ndarray,
    spacing_y_mm: float,
    *,
    fallback_probabilities_rpi: np.ndarray | None = None,
    v5_config: dict[str, Any] | None = None,
) -> HybridV5Result:
    if rootlets_rpi.shape != fallback_segmentation_rpi.shape:
        raise ValueError("Rootlet support and fallback segmentation shapes differ.")
    config = v5_config or {}
    v5 = split_cluster_mean_v5(rootlets_rpi, spacing_y_mm, **config)
    fallback_support = v5.fallback > 0
    classes, recovered = _fallback_classes(
        fallback_segmentation_rpi,
        fallback_support,
        fallback_probabilities_rpi,
    )

    dorsal = v5.dorsal.copy()
    ventral = v5.ventral.copy()
    dorsal[fallback_support & (classes == 1)] = v5.fallback[
        fallback_support & (classes == 1)
    ]
    ventral[fallback_support & (classes == 2)] = v5.fallback[
        fallback_support & (classes == 2)
    ]
    class_map = np.zeros(rootlets_rpi.shape, dtype=np.uint8)
    class_map[dorsal > 0] = 1
    class_map[ventral > 0] = 2

    labels = np.rint(rootlets_rpi).astype(np.int32)
    if np.any((dorsal > 0) & (ventral > 0)):
        raise RuntimeError("Internal error: hybrid dorsal and ventral outputs overlap.")
    if not np.array_equal(dorsal + ventral, labels):
        raise RuntimeError("Internal error: hybrid output changed RootletSeg support or levels.")
    qc = {
        "method": "cluster_mean_v5_plus_voxelwise_fallback",
        "orientation": "RPI",
        "support_preserved": True,
        "fallback_background_voxels_recovered_from_probabilities": recovered,
        "dorsal_voxels": int(np.count_nonzero(dorsal)),
        "ventral_voxels": int(np.count_nonzero(ventral)),
        "v5": v5.qc,
    }
    return HybridV5Result(dorsal, ventral, class_map, qc)


def _to_rpi(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(original, RPI_ORIENTATION)
    return nib.orientations.apply_orientation(data, transform), transform


def _from_rpi(data: np.ndarray, original_affine: np.ndarray) -> np.ndarray:
    original = nib.orientations.io_orientation(original_affine)
    transform = nib.orientations.ornt_transform(RPI_ORIENTATION, original)
    return nib.orientations.apply_orientation(data, transform)


def _save_like(
    data: np.ndarray, reference: nib.Nifti1Image, path: Path, dtype: np.dtype
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(data.astype(dtype), reference.affine, header=header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def _load_probabilities(
    path: Path, image_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    archive = np.load(path)
    for key in ("probabilities", "softmax"):
        if key in archive:
            probabilities = np.asarray(archive[key])
            if image_shape is None or probabilities.shape[1:] == image_shape:
                return probabilities
            # nnU-Net exports probability arrays in the SimpleITK array order
            # (C, Z, Y, X), while nibabel exposes the paired NIfTI as (X, Y, Z).
            if probabilities.shape[1:] == tuple(reversed(image_shape)):
                return probabilities.transpose(0, 3, 2, 1)
            raise ValueError(
                "Fallback probabilities do not match the prediction image in "
                "either X/Y/Z or nnU-Net Z/Y/X order."
            )
    raise ValueError(f"No probabilities or softmax array found in {path}.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rootlet_image = nib.load(args.rootlets)
    fallback_image = nib.load(args.fallback_segmentation)
    if rootlet_image.shape != fallback_image.shape or not np.allclose(
        rootlet_image.affine, fallback_image.affine, atol=1e-4
    ):
        raise ValueError("Rootlet support and fallback prediction must use one grid.")
    rootlets_rpi, transform = _to_rpi(
        np.asanyarray(rootlet_image.dataobj), rootlet_image.affine
    )
    fallback_rpi, fallback_transform = _to_rpi(
        np.asanyarray(fallback_image.dataobj), fallback_image.affine
    )
    if not np.array_equal(transform, fallback_transform):
        raise ValueError("Rootlet and fallback orientation transforms differ.")
    rpi_affine = rootlet_image.affine @ nib.orientations.inv_ornt_aff(
        transform, rootlet_image.shape
    )
    probabilities_rpi = None
    if args.fallback_probabilities:
        probabilities = _load_probabilities(
            Path(args.fallback_probabilities), rootlet_image.shape
        )
        probabilities_rpi = np.stack(
            [nib.orientations.apply_orientation(channel, transform) for channel in probabilities]
        )
    result = combine_v5_with_fallback(
        rootlets_rpi,
        fallback_rpi,
        float(nib.affines.voxel_sizes(rpi_affine)[1]),
        fallback_probabilities_rpi=probabilities_rpi,
    )
    _save_like(
        _from_rpi(result.dorsal, rootlet_image.affine),
        rootlet_image,
        Path(args.output_dorsal),
        np.int16,
    )
    _save_like(
        _from_rpi(result.ventral, rootlet_image.affine),
        rootlet_image,
        Path(args.output_ventral),
        np.int16,
    )
    if args.output_class_map:
        _save_like(
            _from_rpi(result.class_map, rootlet_image.affine),
            rootlet_image,
            Path(args.output_class_map),
            np.uint8,
        )
    qc = dict(result.qc)
    qc.update(
        {
            "rootlets": str(Path(args.rootlets).resolve()),
            "fallback_segmentation": str(Path(args.fallback_segmentation).resolve()),
            "output_dorsal": str(Path(args.output_dorsal).resolve()),
            "output_ventral": str(Path(args.output_ventral).resolve()),
        }
    )
    if args.qc_json:
        path = Path(args.qc_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(qc, indent=2, allow_nan=False) + "\n")
    return qc


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootlets", required=True)
    parser.add_argument("--fallback-segmentation", required=True)
    parser.add_argument("--fallback-probabilities")
    parser.add_argument("--output-dorsal", required=True)
    parser.add_argument("--output-ventral", required=True)
    parser.add_argument("--output-class-map")
    parser.add_argument("--qc-json")
    return parser


def main() -> None:
    print(json.dumps(run(get_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
