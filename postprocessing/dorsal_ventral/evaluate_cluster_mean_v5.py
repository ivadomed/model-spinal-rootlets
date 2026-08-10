#!/usr/bin/env python3
"""Tune V5 on train/validation cases and evaluate once on held-out cases."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from postprocessing.dorsal_ventral.cluster_mean_v5 import split_cluster_mean_v5


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    split: str
    rootlets: np.ndarray
    target: np.ndarray
    spacing_y_mm: float


def _active_crop(rootlets: np.ndarray) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(rootlets > 0)
    lower = np.min(coordinates, axis=0)
    upper = np.max(coordinates, axis=0) + 1
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))  # type: ignore[return-value]


def load_cases(dataset_directory: Path) -> tuple[list[EvaluationCase], dict[str, Any]]:
    manifest = json.loads((dataset_directory / "hybrid_v5_split.json").read_text())
    split_by_case = {
        case_id: split_name
        for split_name, case_ids in manifest["split"].items()
        for case_id in case_ids
    }
    cases: list[EvaluationCase] = []
    for case_id, split_name in split_by_case.items():
        rootlet_image = nib.load(
            dataset_directory / "imagesTr" / f"{case_id}_0001.nii.gz"
        )
        target_image = nib.load(dataset_directory / "labelsTr" / f"{case_id}.nii.gz")
        rootlets = np.rint(np.asanyarray(rootlet_image.dataobj)).astype(np.int16)
        target = np.rint(np.asanyarray(target_image.dataobj)).astype(np.uint8)
        crop = _active_crop(rootlets)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                split=split_name,
                rootlets=rootlets[crop],
                target=target[crop],
                spacing_y_mm=float(nib.affines.voxel_sizes(rootlet_image.affine)[1]),
            )
        )
    return cases, manifest


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_cases(
    cases: Iterable[EvaluationCase], config: dict[str, Any]
) -> dict[str, Any]:
    totals = {
        "levels": 0,
        "deterministic_levels": 0,
        "simple_components": 0,
        "deterministic_simple_components": 0,
        "correct_simple_components": 0,
        "dorsal_deterministic": 0,
        "dorsal_correct": 0,
        "ventral_deterministic": 0,
        "ventral_correct": 0,
        "merged_components": 0,
        "merged_components_routed": 0,
        "deterministic_voxels": 0,
        "correct_deterministic_voxels": 0,
        "support_voxels": 0,
    }
    per_case: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    for case in cases:
        result = split_cluster_mean_v5(
            case.rootlets,
            case.spacing_y_mm,
            **config,
        )
        prediction = np.zeros(case.target.shape, dtype=np.uint8)
        prediction[result.dorsal > 0] = 1
        prediction[result.ventral > 0] = 2
        deterministic = prediction > 0
        case_counts = {key: 0 for key in totals}
        case_counts["levels"] = len(result.qc["levels"])
        case_counts["deterministic_levels"] = result.qc["deterministic_levels"]
        case_counts["support_voxels"] = int(np.count_nonzero(case.rootlets))
        case_counts["deterministic_voxels"] = int(np.count_nonzero(deterministic))
        case_counts["correct_deterministic_voxels"] = int(
            np.count_nonzero(deterministic & (prediction == case.target))
        )

        for level in np.unique(case.rootlets[case.rootlets > 0]):
            component_map, component_count = ndimage.label(
                case.rootlets == level, structure=structure
            )
            for component_id in range(1, int(component_count) + 1):
                component = component_map == component_id
                target_classes = set(int(value) for value in np.unique(case.target[component]))
                is_deterministic = bool(np.all(deterministic[component]))
                if target_classes == {1, 2}:
                    case_counts["merged_components"] += 1
                    if not is_deterministic:
                        case_counts["merged_components_routed"] += 1
                    continue
                if target_classes not in ({1}, {2}):
                    raise ValueError(
                        f"Unexpected target classes {target_classes} in {case.case_id}."
                    )
                truth = next(iter(target_classes))
                case_counts["simple_components"] += 1
                if not is_deterministic:
                    continue
                case_counts["deterministic_simple_components"] += 1
                predicted_class = int(np.bincount(prediction[component], minlength=3).argmax())
                if truth == 1:
                    case_counts["dorsal_deterministic"] += 1
                    case_counts["dorsal_correct"] += int(predicted_class == truth)
                else:
                    case_counts["ventral_deterministic"] += 1
                    case_counts["ventral_correct"] += int(predicted_class == truth)
                case_counts["correct_simple_components"] += int(predicted_class == truth)

        for key in totals:
            totals[key] += int(case_counts[key])
        per_case.append({"case_id": case.case_id, **case_counts})

    dorsal_recall = _safe_ratio(totals["dorsal_correct"], totals["dorsal_deterministic"])
    ventral_recall = _safe_ratio(
        totals["ventral_correct"], totals["ventral_deterministic"]
    )
    balanced_accuracy = (
        float((dorsal_recall + ventral_recall) / 2)
        if dorsal_recall is not None and ventral_recall is not None
        else None
    )
    metrics = {
        **totals,
        "level_coverage": _safe_ratio(
            totals["deterministic_levels"], totals["levels"]
        ),
        "component_coverage": _safe_ratio(
            totals["deterministic_simple_components"], totals["simple_components"]
        ),
        "selective_component_accuracy": _safe_ratio(
            totals["correct_simple_components"],
            totals["deterministic_simple_components"],
        ),
        "selective_component_balanced_accuracy": balanced_accuracy,
        "accuracy_with_routed_as_errors": _safe_ratio(
            totals["correct_simple_components"], totals["simple_components"]
        ),
        "merged_route_recall": _safe_ratio(
            totals["merged_components_routed"], totals["merged_components"]
        ),
        "deterministic_voxel_accuracy": _safe_ratio(
            totals["correct_deterministic_voxels"], totals["deterministic_voxels"]
        ),
        "deterministic_voxel_coverage": _safe_ratio(
            totals["deterministic_voxels"], totals["support_voxels"]
        ),
        "cases": per_case,
    }
    return metrics


def _configurations() -> Iterable[dict[str, Any]]:
    for values in itertools.product(
        (5, 10, 20, 30),
        (0.5, 1.0, 1.5, 2.0),
        (1.25, 1.5, 2.0),
        (0.05, 0.1, 0.15),
        (False, True),
    ):
        yield dict(
            zip(
                (
                    "min_component_voxels",
                    "min_gap_mm",
                    "min_gap_ratio",
                    "crossing_fraction",
                    "allow_two_components",
                ),
                values,
            )
        )


def _metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else -1.0


def tune_and_test(cases: list[EvaluationCase]) -> dict[str, Any]:
    train = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    test = [case for case in cases if case.split == "test"]
    candidates: list[dict[str, Any]] = []
    for config in _configurations():
        train_metrics = evaluate_cases(train, config)
        if _metric(train_metrics, "selective_component_accuracy") < 0.97:
            continue
        if _metric(train_metrics, "merged_route_recall") < 0.90:
            continue
        validation_metrics = evaluate_cases(validation, config)
        candidates.append(
            {
                "config": config,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
    if not candidates:
        raise RuntimeError("No V5 configuration met the training safety constraints.")

    candidates.sort(
        key=lambda item: (
            _metric(item["validation"], "selective_component_balanced_accuracy"),
            _metric(item["validation"], "selective_component_accuracy"),
            min(
                _metric(item["train"], "component_coverage"),
                _metric(item["validation"], "component_coverage"),
            ),
            _metric(item["validation"], "component_coverage"),
        ),
        reverse=True,
    )
    selected = candidates[0]
    test_metrics = evaluate_cases(test, selected["config"])
    return {
        "selection_rule": (
            "train accuracy >=0.97 and merged-route recall >=0.90; then maximize "
            "validation balanced accuracy, accuracy, and minimum train/validation coverage"
        ),
        "searched_configurations": 4 * 4 * 3 * 3 * 2,
        "eligible_configurations": len(candidates),
        "selected_config": selected["config"],
        "train": selected["train"],
        "validation": selected["validation"],
        "test": test_metrics,
        "top_validation_candidates": [
            {
                "config": item["config"],
                "train_component_coverage": item["train"]["component_coverage"],
                "train_selective_accuracy": item["train"][
                    "selective_component_accuracy"
                ],
                "validation_component_coverage": item["validation"][
                    "component_coverage"
                ],
                "validation_selective_accuracy": item["validation"][
                    "selective_component_accuracy"
                ],
            }
            for item in candidates[:10]
        ],
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    cases, split_manifest = load_cases(args.dataset_directory)
    result = tune_and_test(cases)
    result["split_seed"] = split_manifest["split_seed"]
    result["split_counts"] = {
        name: len(case_ids) for name, case_ids in split_manifest["split"].items()
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "selected_config": result["selected_config"],
        "train": {key: result["train"][key] for key in (
            "component_coverage", "selective_component_accuracy", "merged_route_recall"
        )},
        "validation": {key: result["validation"][key] for key in (
            "component_coverage", "selective_component_accuracy"
        )},
        "test": {key: result["test"][key] for key in (
            "component_coverage", "selective_component_accuracy", "merged_route_recall"
        )},
    }, indent=2))


if __name__ == "__main__":
    main()
