#!/usr/bin/env python3
"""Consolidate nnU-Net validation summaries across completed rootlets folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


LEVELS = {
    "2": "C2",
    "3": "C3",
    "4": "C4",
    "5": "C5",
    "6": "C6",
    "7": "C7",
    "8": "C8",
    "9": "T1",
}
EXPECTED_FOLDS = {"0", "1", "2", "3", "4"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fold-summary",
        action="append",
        required=True,
        metavar="FOLD=SUMMARY_JSON",
        help="Repeat for every completed fold.",
    )
    p.add_argument("--checkpoint-label", required=True, choices=("best", "final"))
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def parse_fold_summaries(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        fold, separator, path = value.partition("=")
        if not separator or fold not in EXPECTED_FOLDS or not path:
            raise ValueError(f"Expected FOLD=SUMMARY_JSON with fold 0--4, got: {value}")
        if fold in parsed:
            raise ValueError(f"Fold {fold} was supplied more than once")
        summary = Path(path).resolve()
        if not summary.is_file():
            raise FileNotFoundError(summary)
        parsed[fold] = summary
    return parsed


def case_id(path: str) -> str:
    name = Path(path).name
    return name.removesuffix(".nii.gz").removesuffix(".nii")


def contrast(case: str) -> str:
    if "_inv-1_" in case:
        return "INV1"
    if "_inv-2_" in case:
        return "INV2"
    if case.endswith("_UNIT1"):
        return "UNIT1"
    return "T2w"


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    selected = finite(values)
    return {
        "n": len(selected),
        "mean": statistics.fmean(selected) if selected else math.nan,
        "sd": statistics.stdev(selected) if len(selected) > 1 else math.nan,
        "median": statistics.median(selected) if selected else math.nan,
        "q1": percentile(selected, 0.25),
        "q3": percentile(selected, 0.75),
    }


def binary_dice(prediction: Path, reference: Path) -> float:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Binary Dice requires nibabel and numpy.") from error

    pred_img = nib.load(prediction)
    ref_img = nib.load(reference)
    if pred_img.shape != ref_img.shape or not np.allclose(pred_img.affine, ref_img.affine, atol=1e-4):
        raise ValueError(f"Prediction/reference grid mismatch: {prediction}, {reference}")
    pred = np.asanyarray(pred_img.dataobj) > 0
    ref = np.asanyarray(ref_img.dataobj) > 0
    denominator = int(pred.sum()) + int(ref.sum())
    return 2.0 * int(np.logical_and(pred, ref).sum()) / denominator if denominator else math.nan


def load_rows(fold_summaries: dict[str, Path], checkpoint_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in sorted(fold_summaries, key=int):
        summary = json.loads(fold_summaries[fold].read_text())
        for item in summary["metric_per_case"]:
            case = case_id(item["prediction_file"])
            if case in seen:
                raise ValueError(f"Validation case occurs in multiple supplied folds: {case}")
            seen.add(case)
            metrics = item["metrics"]
            prediction_file = Path(item["prediction_file"])
            if not prediction_file.is_file():
                prediction_file = fold_summaries[fold].parent / prediction_file.name
            reference_file = Path(item["reference_file"])
            if not prediction_file.is_file() or not reference_file.is_file():
                raise FileNotFoundError(
                    f"Missing prediction/reference for {case}: "
                    f"{prediction_file}, {reference_file}"
                )
            level_values: list[float] = []
            row: dict[str, Any] = {
                "checkpoint": checkpoint_label,
                "fold": fold,
                "case_id": case,
                "contrast": contrast(case),
                "prediction_file": str(prediction_file),
                "reference_file": str(reference_file),
            }
            gt_levels: list[str] = []
            pred_levels: list[str] = []
            for label, name in LEVELS.items():
                metric = metrics[label]
                value = float(metric["Dice"])
                row[f"dice_{name}"] = value
                if math.isfinite(value):
                    level_values.append(value)
                if int(metric["n_ref"]) > 0:
                    gt_levels.append(name)
                if int(metric["n_pred"]) > 0:
                    pred_levels.append(name)
            row["macro_level_dice"] = statistics.fmean(level_values) if level_values else math.nan
            row["binary_dice"] = binary_dice(prediction_file, reference_file)
            row["gt_levels"] = ";".join(gt_levels)
            row["pred_levels"] = ";".join(pred_levels)
            rows.append(row)
    return rows


def clean_value(value: Any) -> Any:
    return "" if isinstance(value, float) and not math.isfinite(value) else value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: clean_value(row.get(field, "")) for field in fields} for row in rows)


def summary_rows(
    rows: list[dict[str, Any]],
    group_field: str,
    groups: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in groups:
        selected = rows if group == "All" else [
            row for row in rows if str(row[group_field]) == group
        ]
        macro = stats([float(row["macro_level_dice"]) for row in selected])
        binary = stats([float(row["binary_dice"]) for row in selected])
        output.append(
            {
                group_field: group,
                "n_cases": len(selected),
                **{f"macro_{key}": value for key, value in macro.items()},
                **{f"binary_{key}": value for key, value in binary.items()},
            }
        )
    return output


def level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label, name in LEVELS.items():
        summary = stats([float(row[f"dice_{name}"]) for row in rows])
        output.append(
            {
                "label": label,
                "level": name,
                "n_gt_present": sum(name in row["gt_levels"].split(";") for row in rows),
                "n_pred_present": sum(name in row["pred_levels"].split(";") for row in rows),
                **summary,
            }
        )
    return output


def format_value(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.3f}"


def metric_text(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{format_value(float(row[f'{prefix}_mean']))} ± "
        f"{format_value(float(row[f'{prefix}_sd']))}"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def write_markdown(
    path: Path,
    checkpoint_label: str,
    available: list[str],
    missing: list[str],
    folds: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    levels: list[dict[str, Any]],
) -> None:
    fold_table = markdown_table(
        ["Fold", "Cases", "Macro level Dice", "Binary Dice"],
        [
            [
                str(row["fold"]),
                str(row["n_cases"]),
                metric_text(row, "macro"),
                metric_text(row, "binary"),
            ]
            for row in folds
        ],
    )
    contrast_table = markdown_table(
        ["Contrast", "Cases", "Macro level Dice", "Binary Dice"],
        [
            [
                str(row["contrast"]),
                str(row["n_cases"]),
                metric_text(row, "macro"),
                metric_text(row, "binary"),
            ]
            for row in contrasts
        ],
    )
    level_table = markdown_table(
        ["Level", "GT present", "Predicted", "Valid", "Dice"],
        [
            [
                str(row["level"]),
                str(row["n_gt_present"]),
                str(row["n_pred_present"]),
                str(row["n"]),
                f"{format_value(float(row['mean']))} ± {format_value(float(row['sd']))}",
            ]
            for row in levels
        ],
    )
    path.write_text(
        "\n".join(
            [
                "# Provisional cross-validation metrics",
                "",
                f"- Checkpoint: `{checkpoint_label}`",
                f"- Available folds: {', '.join(available)}",
                f"- Missing folds: {', '.join(missing) if missing else 'none'}",
                "- Status: provisional until all five folds are consolidated.",
                "",
                "## By fold",
                "",
                fold_table,
                "",
                "## By contrast",
                "",
                contrast_table,
                "",
                "## By level",
                "",
                level_table,
                "",
            ]
        )
    )


def main() -> None:
    args = parser().parse_args()
    fold_summaries = parse_fold_summaries(args.fold_summary)
    rows = load_rows(fold_summaries, args.checkpoint_label)
    args.out_dir.mkdir(parents=True, exist_ok=False)

    available = sorted(fold_summaries, key=int)
    missing = sorted(EXPECTED_FOLDS - set(available), key=int)
    folds = summary_rows(rows, "fold", ["All", *available])
    contrasts = summary_rows(rows, "contrast", ["All", "T2w", "INV1", "INV2", "UNIT1"])
    levels = level_rows(rows)

    case_fields = [
        "checkpoint",
        "fold",
        "case_id",
        "contrast",
        "macro_level_dice",
        "binary_dice",
        *[f"dice_{name}" for name in LEVELS.values()],
        "gt_levels",
        "pred_levels",
        "prediction_file",
        "reference_file",
    ]
    summary_fields = [
        "n_cases",
        *[f"macro_{key}" for key in stats([])],
        *[f"binary_{key}" for key in stats([])],
    ]
    write_csv(args.out_dir / "metrics_per_case.csv", rows, case_fields)
    write_csv(args.out_dir / "metrics_by_fold.csv", folds, ["fold", *summary_fields])
    write_csv(args.out_dir / "metrics_by_contrast.csv", contrasts, ["contrast", *summary_fields])
    write_csv(
        args.out_dir / "metrics_by_level.csv",
        levels,
        ["label", "level", "n_gt_present", "n_pred_present", "n", "mean", "sd", "median", "q1", "q3"],
    )
    manifest = {
        "checkpoint": args.checkpoint_label,
        "available_folds": available,
        "missing_folds": missing,
        "n_unique_cases": len(rows),
        "fold_summaries": {fold: str(path) for fold, path in fold_summaries.items()},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_markdown(
        args.out_dir / "summary.md",
        args.checkpoint_label,
        available,
        missing,
        folds,
        contrasts,
        levels,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
