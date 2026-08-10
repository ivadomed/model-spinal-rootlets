#!/usr/bin/env python3
"""Select a V5 fallback architecture from frozen validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else -1.0


def _rank(record: dict[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = record["evaluation"]["hybrid"]
    mean_dice = (
        _value(metrics, "dorsal_dice") + _value(metrics, "ventral_dice")
    ) / 2
    return (
        _value(metrics, "fallback_voxel_accuracy"),
        _value(metrics, "merged_voxel_accuracy"),
        _value(metrics, "voxel_balanced_accuracy"),
        mean_dice,
        record["name"],
    )


def select(candidates: list[tuple[str, Path]]) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("At least two fallback candidates are required.")
    records = []
    for name, path in candidates:
        evaluation = json.loads(path.read_text())
        if evaluation.get("split") != "validation":
            raise ValueError(f"Candidate {name} is not a validation evaluation.")
        records.append(
            {"name": name, "evaluation_json": str(path.resolve()), "evaluation": evaluation}
        )
    winner = max(records, key=_rank)
    return {
        "schema": "rootlet-dv-v5-fallback-selection-v1",
        "split": "validation",
        "rule": [
            "hybrid fallback-region voxel accuracy",
            "hybrid merged-component voxel accuracy",
            "hybrid global voxel balanced accuracy",
            "mean hybrid dorsal/ventral Dice",
            "candidate name as deterministic final tie-break",
        ],
        "selected": winner["name"],
        "candidates": records,
    }


def write_table(selection: dict[str, Any], path: Path) -> None:
    fields = (
        "candidate",
        "selected",
        "fallback_voxel_accuracy",
        "merged_voxel_accuracy",
        "voxel_balanced_accuracy",
        "dorsal_dice",
        "ventral_dice",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in selection["candidates"]:
            metrics = record["evaluation"]["hybrid"]
            writer.writerow(
                {
                    "candidate": record["name"],
                    "selected": record["name"] == selection["selected"],
                    **{field: metrics.get(field) for field in fields[2:]},
                }
            )


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Candidate must use NAME=EVALUATION_JSON.")
    return name, Path(path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, type=_candidate
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = select(args.candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_table(result, args.output_csv)
    print(json.dumps({"selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
