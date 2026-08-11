#!/usr/bin/env python3
"""Constrain the selected V5 3-D network prediction to fixed rootlet support."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.hybrid_v5 import (
    _fallback_classes,
    _load_probabilities,
)


@dataclass(frozen=True)
class ClassifierResult:
    dorsal: np.ndarray
    ventral: np.ndarray
    class_map: np.ndarray
    qc: dict[str, Any]


def classify_rootlet_support(
    rootlets: np.ndarray,
    segmentation: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> ClassifierResult:
    """Partition every input rootlet voxel as dorsal or ventral.

    ``rootlets`` retains the original spinal-level values. ``segmentation`` and
    ``probabilities`` are the raw nnU-Net outputs, with 0=background, 1=dorsal,
    and 2=ventral.
    """

    if rootlets.shape != segmentation.shape:
        raise ValueError("Rootlet support and classifier segmentation shapes differ.")
    if rootlets.ndim != 3:
        raise ValueError("Rootlet support must be a 3-D volume.")
    if not np.all(np.isfinite(rootlets)):
        raise ValueError("Rootlet support contains non-finite values.")
    rounded = np.rint(rootlets)
    if not np.array_equal(rootlets, rounded):
        raise ValueError("Rootlet support must contain integer level labels.")
    if np.any(rounded < 0) or not np.any(rounded > 0):
        raise ValueError("Rootlet support must be non-negative and non-empty.")
    if not np.all(np.isfinite(segmentation)):
        raise ValueError("Classifier segmentation contains non-finite values.")
    rounded_segmentation = np.rint(segmentation)
    if not np.array_equal(segmentation, rounded_segmentation):
        raise ValueError("Classifier segmentation must contain integer class labels.")

    labels = rounded.astype(np.int32)
    support = labels > 0
    classes, recovered = _fallback_classes(
        rounded_segmentation, support, probabilities
    )
    class_map = np.zeros(labels.shape, dtype=np.uint8)
    class_map[support] = classes[support]
    dorsal = np.where(class_map == 1, labels, 0).astype(np.int32)
    ventral = np.where(class_map == 2, labels, 0).astype(np.int32)

    if np.any((dorsal > 0) & (ventral > 0)):
        raise RuntimeError("Internal error: dorsal and ventral outputs overlap.")
    if not np.array_equal(dorsal + ventral, labels):
        raise RuntimeError("Internal error: classifier changed rootlet support or levels.")
    return ClassifierResult(
        dorsal=dorsal,
        ventral=ventral,
        class_map=class_map,
        qc={
            "method": "v5_3d_classifier_support_constrained",
            "orientation": "RPI",
            "support_preserved": True,
            "support_voxels": int(np.count_nonzero(support)),
            "dorsal_voxels": int(np.count_nonzero(dorsal)),
            "ventral_voxels": int(np.count_nonzero(ventral)),
            "background_voxels_recovered_from_probabilities": recovered,
        },
    )


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    rootlet_path = Path(args.rootlets)
    segmentation_path = Path(args.classifier_segmentation)
    rootlet_image = nib.load(rootlet_path)
    segmentation_image = nib.load(segmentation_path)
    if rootlet_image.shape != segmentation_image.shape or not np.allclose(
        rootlet_image.affine, segmentation_image.affine, atol=1e-4
    ):
        raise ValueError("Rootlet support and classifier prediction must use one grid.")
    if tuple(nib.aff2axcodes(rootlet_image.affine)) != ("R", "P", "I"):
        raise ValueError("V5 classifier inputs and outputs must use RPI orientation.")

    rootlets = np.asanyarray(rootlet_image.dataobj)
    segmentation = np.asanyarray(segmentation_image.dataobj)
    probabilities = (
        _load_probabilities(Path(args.classifier_probabilities), rootlet_image.shape)
        if args.classifier_probabilities
        else None
    )
    result = classify_rootlet_support(rootlets, segmentation, probabilities)
    _save_like(result.dorsal, rootlet_image, Path(args.output_dorsal), np.int16)
    _save_like(result.ventral, rootlet_image, Path(args.output_ventral), np.int16)
    _save_like(result.class_map, rootlet_image, Path(args.output_class_map), np.uint8)

    qc = {
        **result.qc,
        "rootlets": str(rootlet_path.resolve()),
        "classifier_segmentation": str(segmentation_path.resolve()),
        "classifier_probabilities": (
            str(Path(args.classifier_probabilities).resolve())
            if args.classifier_probabilities
            else None
        ),
        "output_dorsal": str(Path(args.output_dorsal).resolve()),
        "output_ventral": str(Path(args.output_ventral).resolve()),
        "output_class_map": str(Path(args.output_class_map).resolve()),
    }
    if args.qc_json:
        qc_path = Path(args.qc_json)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(qc, indent=2, allow_nan=False) + "\n")
    return qc


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootlets", required=True)
    parser.add_argument("--classifier-segmentation", required=True)
    parser.add_argument("--classifier-probabilities")
    parser.add_argument("--output-dorsal", required=True)
    parser.add_argument("--output-ventral", required=True)
    parser.add_argument("--output-class-map", required=True)
    parser.add_argument("--qc-json")
    return parser


def main() -> None:
    print(json.dumps(run(get_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
