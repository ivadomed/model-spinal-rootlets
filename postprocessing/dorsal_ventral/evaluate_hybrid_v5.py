#!/usr/bin/env python3
"""Evaluate a trained voxelwise fallback alone and inside the V5 hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from postprocessing.dorsal_ventral.cluster_mean_v5 import split_cluster_mean_v5
from postprocessing.dorsal_ventral.hybrid_v5 import (
    _fallback_classes,
    _load_probabilities,
    combine_v5_with_fallback,
)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _new_counts() -> dict[str, int]:
    return {
        "support": 0,
        "correct": 0,
        "dorsal_truth": 0,
        "dorsal_correct": 0,
        "ventral_truth": 0,
        "ventral_correct": 0,
        "dorsal_prediction": 0,
        "dorsal_intersection": 0,
        "ventral_prediction": 0,
        "ventral_intersection": 0,
        "fallback_support": 0,
        "fallback_correct": 0,
        "simple_components": 0,
        "correct_simple_components": 0,
        "merged_components": 0,
        "merged_support": 0,
        "merged_correct": 0,
    }


def _update_counts(
    counts: dict[str, int],
    truth: np.ndarray,
    prediction: np.ndarray,
    rootlets: np.ndarray,
    fallback_support: np.ndarray,
) -> None:
    support = rootlets > 0
    counts["support"] += int(np.count_nonzero(support))
    counts["correct"] += int(np.count_nonzero(support & (truth == prediction)))
    for label, name in ((1, "dorsal"), (2, "ventral")):
        truth_mask = support & (truth == label)
        prediction_mask = support & (prediction == label)
        counts[f"{name}_truth"] += int(np.count_nonzero(truth_mask))
        counts[f"{name}_correct"] += int(
            np.count_nonzero(truth_mask & prediction_mask)
        )
        counts[f"{name}_prediction"] += int(np.count_nonzero(prediction_mask))
        counts[f"{name}_intersection"] += int(
            np.count_nonzero(truth_mask & prediction_mask)
        )
    counts["fallback_support"] += int(np.count_nonzero(fallback_support))
    counts["fallback_correct"] += int(
        np.count_nonzero(fallback_support & (truth == prediction))
    )

    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    for level in np.unique(rootlets[support]):
        component_map, component_count = ndimage.label(
            rootlets == level, structure=structure
        )
        for component_id in range(1, int(component_count) + 1):
            component = component_map == component_id
            target_classes = set(int(value) for value in np.unique(truth[component]))
            if target_classes == {1, 2}:
                counts["merged_components"] += 1
                counts["merged_support"] += int(np.count_nonzero(component))
                counts["merged_correct"] += int(
                    np.count_nonzero(component & (truth == prediction))
                )
                continue
            counts["simple_components"] += 1
            target_class = next(iter(target_classes))
            predicted_class = int(
                np.bincount(prediction[component], minlength=3).argmax()
            )
            counts["correct_simple_components"] += int(
                predicted_class == target_class
            )


def _summarize(counts: dict[str, int]) -> dict[str, Any]:
    dorsal_recall = _safe_ratio(counts["dorsal_correct"], counts["dorsal_truth"])
    ventral_recall = _safe_ratio(counts["ventral_correct"], counts["ventral_truth"])
    return {
        **counts,
        "voxel_accuracy": _safe_ratio(counts["correct"], counts["support"]),
        "voxel_balanced_accuracy": (
            (dorsal_recall + ventral_recall) / 2
            if dorsal_recall is not None and ventral_recall is not None
            else None
        ),
        "dorsal_dice": _safe_ratio(
            2 * counts["dorsal_intersection"],
            counts["dorsal_truth"] + counts["dorsal_prediction"],
        ),
        "ventral_dice": _safe_ratio(
            2 * counts["ventral_intersection"],
            counts["ventral_truth"] + counts["ventral_prediction"],
        ),
        "fallback_voxel_accuracy": _safe_ratio(
            counts["fallback_correct"], counts["fallback_support"]
        ),
        "simple_component_accuracy": _safe_ratio(
            counts["correct_simple_components"], counts["simple_components"]
        ),
        "merged_voxel_accuracy": _safe_ratio(
            counts["merged_correct"], counts["merged_support"]
        ),
    }


def evaluate(
    dataset_directory: Path,
    prediction_directory: Path,
    split_name: str,
    output_directory: Path,
) -> dict[str, Any]:
    manifest = json.loads((dataset_directory / "fallback_manifest.json").read_text())
    case_ids = manifest["split"][split_name]
    image_folder = "imagesTr" if split_name in {"train", "validation"} else "imagesTs"
    label_folder = "labelsTr" if split_name in {"train", "validation"} else "labelsTs"
    hybrid_counts = _new_counts()
    model_counts = _new_counts()
    per_case: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        rootlet_image = nib.load(
            dataset_directory / image_folder / f"{case_id}_0001.nii.gz"
        )
        rootlets = np.rint(np.asanyarray(rootlet_image.dataobj)).astype(np.int16)
        target = np.rint(
            np.asanyarray(
                nib.load(dataset_directory / label_folder / f"{case_id}.nii.gz").dataobj
            )
        ).astype(np.uint8)
        segmentation = np.rint(
            np.asanyarray(
                nib.load(prediction_directory / f"{case_id}.nii.gz").dataobj
            )
        ).astype(np.uint8)
        probability_path = prediction_directory / f"{case_id}.npz"
        probabilities = (
            _load_probabilities(probability_path, rootlet_image.shape)
            if probability_path.is_file()
            else None
        )
        spacing_y_mm = float(nib.affines.voxel_sizes(rootlet_image.affine)[1])
        v5 = split_cluster_mean_v5(rootlets, spacing_y_mm)
        model_classes, _recovered = _fallback_classes(
            segmentation, rootlets > 0, probabilities
        )
        hybrid = combine_v5_with_fallback(
            rootlets,
            segmentation,
            spacing_y_mm,
            fallback_probabilities_rpi=probabilities,
        )
        case_hybrid = _new_counts()
        case_model = _new_counts()
        fallback_support = v5.fallback > 0
        _update_counts(
            case_hybrid, target, hybrid.class_map, rootlets, fallback_support
        )
        _update_counts(case_model, target, model_classes, rootlets, fallback_support)
        for key in hybrid_counts:
            hybrid_counts[key] += case_hybrid[key]
            model_counts[key] += case_model[key]
        per_case.append(
            {
                "case_id": case_id,
                "hybrid": _summarize(case_hybrid),
                "model_only": _summarize(case_model),
                "v5_level_coverage": v5.qc["deterministic_levels"]
                / len(v5.qc["levels"]),
            }
        )
        header = rootlet_image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image(
                hybrid.class_map.astype(np.uint8), rootlet_image.affine, header=header
            ),
            output_directory / f"{case_id}_desc-hybridV5_dseg.nii.gz",
        )

    return {
        "schema": "rootlet-dv-hybrid-v5-evaluation-v1",
        "split": split_name,
        "case_count": len(case_ids),
        "hybrid": _summarize(hybrid_counts),
        "model_only": _summarize(model_counts),
        "cases": per_case,
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--prediction-directory", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = evaluate(
        args.dataset_directory,
        args.prediction_directory,
        args.split,
        args.output_directory,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"hybrid": result["hybrid"], "model_only": result["model_only"]}, indent=2))


if __name__ == "__main__":
    main()
