#!/usr/bin/env python3
"""Write a participant-free aggregate summary of hybrid V5 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EVALUATION_METRICS = (
    "support",
    "correct",
    "fallback_support",
    "fallback_correct",
    "simple_components",
    "correct_simple_components",
    "merged_components",
    "merged_support",
    "merged_correct",
    "voxel_accuracy",
    "voxel_balanced_accuracy",
    "dorsal_dice",
    "ventral_dice",
    "fallback_voxel_accuracy",
    "simple_component_accuracy",
    "merged_voxel_accuracy",
)


def _evaluation(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in EVALUATION_METRICS}


def summarize(
    deterministic_json: Path,
    selection_json: Path,
    test_json: Path,
    external_json: Path,
    runtime_json: Path | None = None,
) -> dict[str, Any]:
    deterministic = json.loads(deterministic_json.read_text())
    selection = json.loads(selection_json.read_text())
    test = json.loads(test_json.read_text())
    external = json.loads(external_json.read_text())
    deterministic_test = {
        key: value for key, value in deterministic["test"].items() if key != "cases"
    }
    validation = {}
    for candidate in selection["candidates"]:
        validation[candidate["name"]] = _evaluation(
            candidate["evaluation"]["hybrid"]
        )
    datasets = {}
    for dataset in sorted(set(case["dataset"] for case in external["cases"])):
        cases = [case for case in external["cases"] if case["dataset"] == dataset]
        datasets[dataset] = {
            "cases": len(cases),
            "v5_level_coverage_mean": float(
                np.mean([case["v5_level_coverage"] for case in cases])
            ),
            "v5_level_coverage_range": [
                float(min(case["v5_level_coverage"] for case in cases)),
                float(max(case["v5_level_coverage"] for case in cases)),
            ],
            "fallback_voxel_fraction_mean": float(
                np.mean([case["fallback_voxel_fraction"] for case in cases])
            ),
            "dorsal_fraction_mean": float(
                np.mean([case["dorsal_fraction"] for case in cases])
            ),
            "exact_support_partitions": sum(
                case["support_preserved"] for case in cases
            ),
        }
    summary = {
        "schema": "rootlet-dv-hybrid-v5-public-summary-v1",
        "data": {
            "expert_cases": 15,
            "train_cases": 10,
            "validation_cases": 2,
            "test_cases": 3,
            "expert_source": "HC-Leipzig UNIT1",
            "quantitative_rootlet_input": "manual level-labelled rootlet masks",
            "external_rootlet_input": "RootletSeg predictions",
        },
        "deterministic_v5_test": deterministic_test,
        "fallback_selection": {
            "split": selection["split"],
            "rule": selection["rule"],
            "selected": selection["selected"],
            "candidates": validation,
        },
        "heldout_test": {
            "case_count": test["case_count"],
            "hybrid": _evaluation(test["hybrid"]),
            "fallback_alone": _evaluation(test["model_only"]),
        },
        "external_qc": {
            "case_count": external["case_count"],
            "exact_support_partitions": external["exact_support_partitions"],
            "datasets": datasets,
            "interpretation": "No expert external D/V ground truth; not accuracy.",
        },
    }
    if runtime_json is not None:
        summary["runtime"] = json.loads(runtime_json.read_text())
    return summary


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-json", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--test-json", required=True, type=Path)
    parser.add_argument("--external-json", required=True, type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = summarize(
        args.deterministic_json,
        args.selection_json,
        args.test_json,
        args.external_json,
        args.runtime_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
