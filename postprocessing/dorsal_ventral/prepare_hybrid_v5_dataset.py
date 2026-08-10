#!/usr/bin/env python3
"""Freeze complete expert D/V cases and create a leakage-safe split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from postprocessing.dorsal_ventral.labeling_app_core import (
    MERGED_DV_CLASS,
    discover_reference_label_cases,
    export_nnunet_case,
    load_case,
    merge_saved_annotations,
    read_review_csv,
    write_nnunet_dataset_json,
)


EXPERT_LABELS = {"dorsal", "ventral", MERGED_DV_CLASS}


def _review_path(annotation_directory: Path, case_id: str) -> Path:
    return annotation_directory / f"{case_id}_desc-rootlet-cluster_review.csv"


def _complete_rows(path: Path) -> list[dict[str, str]] | None:
    rows = read_review_csv(path)
    if not rows or any(row.get("expert_class") not in EXPERT_LABELS for row in rows):
        return None
    return rows


def _split_cases(
    case_ids: list[str],
    *,
    seed: str,
    train_cases: int,
    validation_cases: int,
    test_cases: int,
) -> dict[str, list[str]]:
    expected = train_cases + validation_cases + test_cases
    if len(case_ids) != expected:
        raise ValueError(f"Expected {expected} complete cases, found {len(case_ids)}.")
    ranked = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest(),
    )
    train_stop = train_cases
    validation_stop = train_stop + validation_cases
    return {
        "train": ranked[:train_stop],
        "validation": ranked[train_stop:validation_stop],
        "test": ranked[validation_stop:],
    }


def prepare_dataset(
    dataset_root: Path,
    annotation_directory: Path,
    output_directory: Path,
    *,
    seed: str = "dv-v5-2026-08-10",
    train_cases: int = 10,
    validation_cases: int = 2,
    test_cases: int = 3,
) -> dict[str, Any]:
    sources = discover_reference_label_cases(dataset_root)
    complete_sources = []
    row_counts: dict[str, dict[str, int]] = {}
    for source in sources:
        rows = _complete_rows(_review_path(annotation_directory, source.case_id))
        if rows is None:
            continue
        complete_sources.append(source)
        row_counts[source.case_id] = {
            "components": len(rows),
            "dorsal": sum(row["expert_class"] == "dorsal" for row in rows),
            "ventral": sum(row["expert_class"] == "ventral" for row in rows),
            "merged_dv": sum(row["expert_class"] == MERGED_DV_CLASS for row in rows),
        }

    expected_cases = train_cases + validation_cases + test_cases
    if len(complete_sources) < expected_cases:
        raise ValueError(
            f"Need {expected_cases} complete cases; found {len(complete_sources)}."
        )
    if len(complete_sources) > expected_cases:
        complete_sources = complete_sources[:expected_cases]

    split = _split_cases(
        [source.case_id for source in complete_sources],
        seed=seed,
        train_cases=train_cases,
        validation_cases=validation_cases,
        test_cases=test_cases,
    )
    split_by_case = {
        case_id: split_name
        for split_name, case_ids in split.items()
        for case_id in case_ids
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, str]] = []
    for source in complete_sources:
        case = load_case(
            source.anatomy_path,
            source.rootlets_path,
            source.cord_path,
            case_id=source.case_id,
        )
        saved_rows = read_review_csv(
            _review_path(annotation_directory, source.case_id)
        )
        records = merge_saved_annotations(case.records, saved_rows)
        entry = export_nnunet_case(case, output_directory, records)
        entry["split"] = split_by_case[source.case_id]
        exported.append(entry)

    dataset_json = write_nnunet_dataset_json(output_directory)
    manifest = {
        "schema": "rootlet-dorsal-ventral-hybrid-v5-split-v1",
        "split_seed": seed,
        "orientation": "RPI",
        "input_channels": {"0": "MRI", "1": "level-labelled rootlet support"},
        "target_labels": {"background": 0, "dorsal": 1, "ventral": 2},
        "split": split,
        "counts": {
            "train": train_cases,
            "validation": validation_cases,
            "test": test_cases,
            "components": sum(item["components"] for item in row_counts.values()),
            "merged_dv": sum(item["merged_dv"] for item in row_counts.values()),
        },
        "case_counts": row_counts,
        "dataset_json": str(dataset_json.resolve()),
        "cases": exported,
    }
    manifest_path = output_directory / "hybrid_v5_split.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--annotation-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--seed", default="dv-v5-2026-08-10")
    parser.add_argument("--train-cases", type=int, default=10)
    parser.add_argument("--validation-cases", type=int, default=2)
    parser.add_argument("--test-cases", type=int, default=3)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    manifest = prepare_dataset(
        args.dataset_root,
        args.annotation_directory,
        args.output_directory,
        seed=args.seed,
        train_cases=args.train_cases,
        validation_cases=args.validation_cases,
        test_cases=args.test_cases,
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
