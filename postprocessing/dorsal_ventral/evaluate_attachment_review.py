#!/usr/bin/env python3
"""Score deterministic attachment classes after an expert fills review CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CLASSES = ("dorsal", "ventral")
PREDICTIONS = (*CLASSES, "unclear")
EXPERT_CLASSES = (*CLASSES, "unclear")


def read_reviews(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    required = {"subject", "attachment_id", "level", "side", "predicted_class", "expert_class"}
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for raw in reader:
                row = {key: (value or "").strip().lower() for key, value in raw.items()}
                key = (str(path.resolve()), row["subject"], row["attachment_id"])
                if key in seen:
                    raise ValueError(f"Duplicate attachment row: {row['subject']} {row['attachment_id']}")
                seen.add(key)
                if row["predicted_class"] not in PREDICTIONS:
                    raise ValueError(f"Invalid predicted_class: {row['predicted_class']}")
                if row["expert_class"] and row["expert_class"] not in EXPERT_CLASSES:
                    raise ValueError(f"Invalid expert_class: {row['expert_class']}")
                row["source_csv"] = str(path.resolve())
                rows.append(row)
    if not rows:
        raise ValueError("No attachment review rows found.")
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def score_rows(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    rows = list(rows)
    reviewed = [row for row in rows if row["expert_class"] in EXPERT_CLASSES]
    labeled = [row for row in rows if row["expert_class"] in CLASSES]
    confusion = {
        truth: {prediction: 0 for prediction in PREDICTIONS} for truth in CLASSES
    }
    for row in labeled:
        confusion[row["expert_class"]][row["predicted_class"]] += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in CLASSES:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label][prediction] for prediction in PREDICTIONS if prediction != label)
        false_positive = sum(confusion[truth][label] for truth in CLASSES if truth != label)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        per_class[label] = {
            "support": true_positive + false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    recalls = [record["recall"] for record in per_class.values() if record["recall"] is not None]
    f1_values = [record["f1"] for record in per_class.values() if record["f1"] is not None]
    correct = sum(confusion[label][label] for label in CLASSES)
    non_abstained = sum(
        confusion[truth][prediction] for truth in CLASSES for prediction in CLASSES
    )
    return {
        "review_rows": len(rows),
        "expert_reviewed": len(reviewed),
        "expert_scorable": len(labeled),
        "expert_review_coverage": _safe_ratio(len(reviewed), len(rows)),
        "expert_scorable_coverage": _safe_ratio(len(labeled), len(rows)),
        "prediction_coverage": _safe_ratio(non_abstained, len(labeled)),
        "accuracy_with_abstentions_as_errors": _safe_ratio(correct, len(labeled)),
        "selective_accuracy_non_abstained": _safe_ratio(correct, non_abstained),
        "balanced_accuracy_with_abstentions_as_errors": (
            sum(recalls) / len(recalls) if len(recalls) == len(CLASSES) else None
        ),
        "macro_f1_with_abstentions_as_errors": (
            sum(f1_values) / len(f1_values) if len(f1_values) == len(CLASSES) else None
        ),
        "per_class": per_class,
        "confusion": confusion,
    }


def evaluate(paths: Iterable[Path]) -> dict[str, Any]:
    rows = read_reviews(paths)
    labeled = [row for row in rows if row["expert_class"] in CLASSES]
    group_fields = ("subject", "level", "side")
    grouped: dict[str, dict[str, Any]] = {}
    for field in group_fields:
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in labeled:
            buckets[row[field]].append(row)
        grouped[field] = {key: score_rows(value) for key, value in sorted(buckets.items())}
    return {
        "scope": (
            "branch-attachment classification; unclear predictions are abstentions and count "
            "as errors in balanced accuracy and macro-F1"
        ),
        "overall": score_rows(rows),
        "by": grouped,
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    result = evaluate(Path(path) for path in args.review_csv)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
