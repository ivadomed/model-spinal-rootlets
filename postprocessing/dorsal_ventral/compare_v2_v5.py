#!/usr/bin/env python3
"""Compare V2-V5 on one frozen expert-labelled split.

Accuracy is measured only inside the fixed level-labelled rootlet support.
V2-V4 share one spinal-cord mask. V5 reports both its selective deterministic
gate and the complete gate-plus-network output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.cluster_mean_v5 import split_cluster_mean_v5
from postprocessing.dorsal_ventral.evaluate_cluster_mean_v5 import (
    EvaluationCase,
    evaluate_cases,
)
from postprocessing.dorsal_ventral.evaluate_hybrid_v5 import (
    _fallback_classes,
    _new_counts,
    _summarize,
    _update_counts,
)
from postprocessing.dorsal_ventral.hybrid_v5 import (
    _load_probabilities,
    combine_v5_with_fallback,
)
from postprocessing.dorsal_ventral.split_rootlets import (
    _from_canonical,
    _to_canonical,
    split_rootlets,
)


ATTACHMENT_METHODS = {
    "V2": "attachment_island",
    "V3": "paired_attachment",
    "V4": "graph_attachment",
}
METHOD_ORDER = ("V2", "V3", "V4", "V5 hybrid", "3-D classifier")


def _strict_labels(image: nib.Nifti1Image, name: str) -> np.ndarray:
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains non-finite values.")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-4):
        raise ValueError(f"{name} must be integer-valued.")
    return rounded.astype(np.int16)


def _same_grid(candidate: nib.Nifti1Image, reference: nib.Nifti1Image) -> bool:
    return candidate.shape == reference.shape and np.allclose(
        candidate.affine, reference.affine, atol=1e-4
    )


def _class_map(dorsal: np.ndarray, ventral: np.ndarray) -> np.ndarray:
    result = np.zeros(dorsal.shape, dtype=np.uint8)
    result[dorsal > 0] = 1
    result[ventral > 0] = 2
    return result


def _repeat(call: Callable[[], Any], count: int) -> tuple[Any, list[float]]:
    result = None
    durations: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        result = call()
        durations.append(time.perf_counter() - started)
    return result, durations


def _timing(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"runs": 0, "median_seconds": None, "range_seconds": [None, None]}
    return {
        "runs": len(values),
        "median_seconds": float(median(values)),
        "range_seconds": [float(min(values)), float(max(values))],
        "all_seconds": [float(value) for value in values],
    }


def _save_class_map(
    data: np.ndarray, reference: nib.Nifti1Image, path: Path
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(data.astype(np.uint8), reference.affine, header=header),
        path,
    )


def _full_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    rootlets: np.ndarray,
    fallback_support: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    counts = _new_counts()
    if fallback_support is None:
        fallback_support = np.zeros(rootlets.shape, dtype=bool)
    _update_counts(
        counts,
        target,
        prediction,
        rootlets,
        fallback_support,
    )
    metrics = _summarize(counts)
    flipped = prediction.copy()
    flipped[prediction == 1] = 2
    flipped[prediction == 2] = 1
    support = rootlets > 0
    metrics["flipped_voxel_accuracy"] = float(
        np.mean(flipped[support] == target[support])
    )
    return metrics, counts


def _add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for key in total:
        total[key] += current[key]


def _elapsed_seconds(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    if not match:
        raise ValueError(f"Cannot find GNU time elapsed duration in {path}.")
    fields = [float(value) for value in match.group(1).split(":")]
    seconds = 0.0
    for value in fields:
        seconds = seconds * 60.0 + value
    return seconds


def _copy_case_inputs(
    alias_directory: Path,
    anatomy: nib.Nifti1Image,
    rootlet_image: nib.Nifti1Image,
    target_image: nib.Nifti1Image,
    cord_image: nib.Nifti1Image,
) -> None:
    alias_directory.mkdir(parents=True, exist_ok=True)
    nib.save(anatomy, alias_directory / "anatomy.nii.gz")
    nib.save(rootlet_image, alias_directory / "rootlets.nii.gz")
    nib.save(target_image, alias_directory / "expert_dseg.nii.gz")
    nib.save(cord_image, alias_directory / "cord.nii.gz")


def compare(
    dataset_directory: Path,
    cord_directory: Path,
    prediction_directory: Path,
    output_directory: Path,
    *,
    repeats: int,
    network_batch_seconds: float,
    cord_time_directory: Path | None,
    cord_time_tag: str,
    deterministic_hardware: str,
    primary_network_label: str,
    recovery_network_batch_seconds: float | None,
    recovery_network_label: str | None,
    cord_method_label: str,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    manifest = json.loads((dataset_directory / "fallback_manifest.json").read_text())
    case_ids = manifest["split"]["test"]
    totals = {name: _new_counts() for name in METHOD_ORDER}
    method_times = {name: [] for name in (*ATTACHMENT_METHODS, "V5 gate", "V5 combine")}
    gate_cases: list[EvaluationCase] = []
    per_case: list[dict[str, Any]] = []
    cord_times: list[float] = []

    for index, case_id in enumerate(case_ids):
        alias = f"Case {chr(ord('A') + index)}"
        alias_slug = alias.replace(" ", "_")
        anatomy_image = nib.load(
            dataset_directory / "imagesTs" / f"{case_id}_0000.nii.gz"
        )
        rootlet_image = nib.load(
            dataset_directory / "imagesTs" / f"{case_id}_0001.nii.gz"
        )
        target_image = nib.load(
            dataset_directory / "labelsTs" / f"{case_id}.nii.gz"
        )
        cord_image = nib.load(cord_directory / f"{case_id}_label-SC_seg.nii.gz")
        prediction_image = nib.load(prediction_directory / f"{case_id}.nii.gz")
        for name, image in (
            ("rootlets", rootlet_image),
            ("target", target_image),
            ("cord", cord_image),
            ("network prediction", prediction_image),
        ):
            if not _same_grid(image, anatomy_image):
                raise ValueError(f"{alias} {name} grid differs from the anatomy.")

        rootlets = _strict_labels(rootlet_image, f"{alias} rootlets")
        target = _strict_labels(target_image, f"{alias} target").astype(np.uint8)
        cord = _strict_labels(cord_image, f"{alias} cord")
        segmentation = _strict_labels(
            prediction_image, f"{alias} network prediction"
        ).astype(np.uint8)
        if set(np.unique(target[rootlets > 0])) - {1, 2}:
            raise ValueError(f"{alias} target must be dorsal=1 and ventral=2.")
        if not np.array_equal(target > 0, rootlets > 0):
            raise ValueError(f"{alias} target does not match the rootlet support.")
        if tuple(nib.aff2axcodes(rootlet_image.affine)) != ("R", "P", "I"):
            raise ValueError(f"{alias} rootlet input is not RPI.")

        alias_directory = output_directory / alias_slug
        _copy_case_inputs(
            alias_directory,
            anatomy_image,
            rootlet_image,
            target_image,
            cord_image,
        )
        case_record: dict[str, Any] = {"case": alias, "methods": {}}

        rootlet_ras, rootlet_transform = _to_canonical(
            rootlets, rootlet_image.affine
        )
        cord_ras, cord_transform = _to_canonical(cord, cord_image.affine)
        if not np.array_equal(rootlet_transform, cord_transform):
            raise ValueError(f"{alias} cord and rootlet transforms differ.")
        canonical_affine = rootlet_image.affine @ nib.orientations.inv_ornt_aff(
            rootlet_transform, rootlet_image.shape
        )
        spacing = tuple(
            float(value) for value in nib.affines.voxel_sizes(canonical_affine)
        )
        for name, strategy in ATTACHMENT_METHODS.items():
            result, durations = _repeat(
                lambda strategy=strategy: split_rootlets(
                    rootlet_ras,
                    cord_ras,
                    spacing,
                    affine=canonical_affine,
                    seed_strategy=strategy,
                ),
                repeats,
            )
            method_times[name].extend(durations)
            dorsal = _from_canonical(result.dorsal, rootlet_image.affine)
            ventral = _from_canonical(result.ventral, rootlet_image.affine)
            prediction = _class_map(dorsal, ventral)
            metrics, counts = _full_metrics(target, prediction, rootlets)
            _add_counts(totals[name], counts)
            case_record["methods"][name] = metrics
            _save_class_map(
                prediction, rootlet_image, alias_directory / f"{name}_dseg.nii.gz"
            )

        spacing_y_mm = float(nib.affines.voxel_sizes(rootlet_image.affine)[1])
        v5_gate, gate_durations = _repeat(
            lambda: split_cluster_mean_v5(rootlets, spacing_y_mm), repeats
        )
        method_times["V5 gate"].extend(gate_durations)
        gate_prediction = _class_map(v5_gate.dorsal, v5_gate.ventral)
        gate_cases.append(
            EvaluationCase(
                case_id=alias,
                split="test",
                rootlets=rootlets,
                target=target,
                spacing_y_mm=spacing_y_mm,
            )
        )
        case_record["v5_gate"] = {
            "level_coverage": v5_gate.qc["deterministic_levels"]
            / len(v5_gate.qc["levels"]),
            "voxel_coverage": float(np.mean(gate_prediction[rootlets > 0] > 0)),
            "selective_voxel_accuracy": float(
                np.mean(
                    gate_prediction[gate_prediction > 0]
                    == target[gate_prediction > 0]
                )
            ),
        }
        _save_class_map(
            gate_prediction,
            rootlet_image,
            alias_directory / "V5_gate_dseg.nii.gz",
        )

        probability_path = prediction_directory / f"{case_id}.npz"
        probabilities = _load_probabilities(probability_path, rootlet_image.shape)
        model_classes, _ = _fallback_classes(
            segmentation, rootlets > 0, probabilities
        )
        model_metrics, model_counts = _full_metrics(
            target, model_classes, rootlets, v5_gate.fallback > 0
        )
        _add_counts(totals["3-D classifier"], model_counts)
        case_record["methods"]["3-D classifier"] = model_metrics
        _save_class_map(
            model_classes,
            rootlet_image,
            alias_directory / "3-D_classifier_dseg.nii.gz",
        )

        hybrid, combine_durations = _repeat(
            lambda: combine_v5_with_fallback(
                rootlets,
                segmentation,
                spacing_y_mm,
                fallback_probabilities_rpi=probabilities,
            ),
            repeats,
        )
        method_times["V5 combine"].extend(combine_durations)
        hybrid_metrics, hybrid_counts = _full_metrics(
            target, hybrid.class_map, rootlets, v5_gate.fallback > 0
        )
        _add_counts(totals["V5 hybrid"], hybrid_counts)
        case_record["methods"]["V5 hybrid"] = hybrid_metrics
        _save_class_map(
            hybrid.class_map,
            rootlet_image,
            alias_directory / "V5_hybrid_dseg.nii.gz",
        )
        per_case.append(case_record)

        if cord_time_directory is not None:
            cord_times.append(
                _elapsed_seconds(
                    cord_time_directory / f"{case_id}_{cord_time_tag}.time"
                )
            )

    methods: dict[str, Any] = {}
    for name in METHOD_ORDER:
        methods[name] = {"metrics": _summarize(totals[name])}
        if name in ATTACHMENT_METHODS:
            methods[name]["algorithm_timing"] = _timing(method_times[name])
            methods[name]["requires_cord_mask"] = True
        elif name == "3-D classifier":
            methods[name]["network_batch_seconds"] = float(network_batch_seconds)
            methods[name]["network_benchmark"] = primary_network_label
            methods[name]["seconds_per_case"] = float(
                network_batch_seconds / len(case_ids)
            )
            methods[name]["requires_cord_mask"] = False
        else:
            combine = _timing(method_times["V5 combine"])
            methods[name]["combine_timing"] = combine
            methods[name]["network_batch_seconds"] = float(network_batch_seconds)
            methods[name]["network_benchmark"] = primary_network_label
            methods[name]["seconds_per_case"] = float(
                network_batch_seconds / len(case_ids)
                + combine["median_seconds"]
            )
            methods[name]["requires_cord_mask"] = False

    gate_metrics = evaluate_cases(gate_cases, {})
    gate_metrics.pop("cases", None)
    result = {
        "schema": "rootlet-dv-v2-v5-heldout-comparison-v1",
        "split": "frozen test",
        "split_seed": manifest["split_seed"],
        "case_count": len(case_ids),
        "orientation": "RPI input; V2-V4 internally use canonical RAS",
        "accuracy_scope": "manual rootlet voxels only; background excluded",
        "rootlet_input": "manual level-labelled rootlet masks",
        "cord_input": "one shared automatic SCT spinal-cord mask for V2-V4",
        "timing_scope": {
            "V2-V4_and_V5_gate": (
                f"arrays already loaded on {deterministic_hardware}"
            ),
            "V5_network": "wall time includes preprocessing and probability export",
            "cord": "cropping, SCT inference, and restoration to the full grid",
        },
        "network_benchmarks": [
            {
                "label": primary_network_label,
                "batch_seconds": float(network_batch_seconds),
                "case_count": len(case_ids),
                "seconds_per_case": float(network_batch_seconds / len(case_ids)),
                "used_in_method_table": True,
            },
            *(
                [
                    {
                        "label": recovery_network_label,
                        "batch_seconds": float(recovery_network_batch_seconds),
                        "case_count": len(case_ids),
                        "seconds_per_case": float(
                            recovery_network_batch_seconds / len(case_ids)
                        ),
                        "used_in_method_table": False,
                    }
                ]
                if recovery_network_batch_seconds is not None
                and recovery_network_label is not None
                else []
            ),
        ],
        "method_order": list(METHOD_ORDER),
        "methods": methods,
        "v5_gate": {
            "metrics": gate_metrics,
            "algorithm_timing": _timing(method_times["V5 gate"]),
            "requires_cord_mask": False,
            "interpretation": (
                "Selective method; accuracy applies only where V5 accepted a level."
            ),
        },
        "cord_segmentation": (
            {
                **_timing(cord_times),
                "shared_by": ["V2", "V3", "V4"],
                "method": cord_method_label,
                "included_in_algorithm_timing": False,
            }
            if cord_times
            else None
        ),
        "cases": per_case,
        "limits": [
            "Three held-out cases only.",
            "One expert rater.",
            (
                "Manual rootlet masks, so accuracy is conditional on correct "
                "rootlet support."
            ),
            "V2-V4 depend on an automatically generated cord mask; V5 does not.",
        ],
    }
    return result


def write_csv(result: dict[str, Any], path: Path) -> None:
    fields = (
        "method",
        "accuracy_scope",
        "voxel_accuracy",
        "balanced_accuracy",
        "dorsal_dice",
        "ventral_dice",
        "component_accuracy",
        "merged_accuracy",
        "voxel_coverage",
        "component_coverage",
        "seconds_per_case",
        "timing_scope",
        "requires_cord_mask",
        "prerequisite",
        "prerequisite_seconds_per_case",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in result["method_order"]:
            record = result["methods"][name]
            metrics = record["metrics"]
            timing = record.get("algorithm_timing", {})
            if name in ATTACHMENT_METHODS:
                timing_scope = result["timing_scope"]["V2-V4_and_V5_gate"]
            elif name == "3-D classifier":
                timing_scope = record["network_benchmark"]
            else:
                timing_scope = (
                    f"{record['network_benchmark']} + "
                    f"{result['timing_scope']['V2-V4_and_V5_gate']} combine"
                )
            cord = result.get("cord_segmentation")
            writer.writerow(
                {
                    "method": name,
                    "accuracy_scope": "all manual rootlet voxels",
                    "voxel_accuracy": metrics["voxel_accuracy"],
                    "balanced_accuracy": metrics["voxel_balanced_accuracy"],
                    "dorsal_dice": metrics["dorsal_dice"],
                    "ventral_dice": metrics["ventral_dice"],
                    "component_accuracy": metrics["simple_component_accuracy"],
                    "merged_accuracy": metrics["merged_voxel_accuracy"],
                    "seconds_per_case": record.get(
                        "seconds_per_case", timing.get("median_seconds")
                    ),
                    "timing_scope": timing_scope,
                    "requires_cord_mask": record["requires_cord_mask"],
                    "prerequisite": (
                        "automatic cropped SCT cord mask"
                        if record["requires_cord_mask"]
                        else ""
                    ),
                    "prerequisite_seconds_per_case": (
                        cord["median_seconds"]
                        if record["requires_cord_mask"] and cord
                        else ""
                    ),
                }
            )
        gate = result["v5_gate"]
        writer.writerow(
            {
                "method": "V5 gate (selective)",
                "accuracy_scope": "accepted voxels/components only",
                "voxel_accuracy": gate["metrics"]["deterministic_voxel_accuracy"],
                "component_accuracy": gate["metrics"][
                    "selective_component_accuracy"
                ],
                "voxel_coverage": gate["metrics"]["deterministic_voxel_coverage"],
                "component_coverage": gate["metrics"]["component_coverage"],
                "seconds_per_case": gate["algorithm_timing"]["median_seconds"],
                "timing_scope": result["timing_scope"]["V2-V4_and_V5_gate"],
                "requires_cord_mask": False,
            }
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--cord-directory", required=True, type=Path)
    parser.add_argument("--prediction-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--network-batch-seconds", type=float, required=True)
    parser.add_argument("--cord-time-directory", type=Path)
    parser.add_argument("--cord-time-tag", default="cord_v2")
    parser.add_argument("--deterministic-hardware", default="current CPU")
    parser.add_argument(
        "--primary-network-label", default="locked primary GPU benchmark"
    )
    parser.add_argument("--recovery-network-batch-seconds", type=float)
    parser.add_argument("--recovery-network-label")
    parser.add_argument(
        "--cord-method-label",
        default="automatic SCT spinal-cord segmentation",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = compare(
        args.dataset_directory,
        args.cord_directory,
        args.prediction_directory,
        args.output_directory,
        repeats=args.repeats,
        network_batch_seconds=args.network_batch_seconds,
        cord_time_directory=args.cord_time_directory,
        cord_time_tag=args.cord_time_tag,
        deterministic_hardware=args.deterministic_hardware,
        primary_network_label=args.primary_network_label,
        recovery_network_batch_seconds=args.recovery_network_batch_seconds,
        recovery_network_label=args.recovery_network_label,
        cord_method_label=args.cord_method_label,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(result, args.output_csv)
    print(
        json.dumps(
            {"methods": result["methods"], "v5_gate": result["v5_gate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
