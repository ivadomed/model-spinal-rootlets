#!/usr/bin/env python3
"""Compute NaN-safe, per-level Dice summaries for a canonical test set."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np


LEVELS = {
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    8: "C8",
    9: "T1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True,
                        help="TSV containing case_id, contrast, and reference columns.")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def group_for_contrast(contrast: str) -> str:
    if contrast == "T2w":
        return "T2w"
    if contrast == "UNIT1":
        return "UNIT1"
    if contrast.startswith("inv-1"):
        return "INV1"
    if contrast.startswith("inv-2"):
        return "INV2"
    raise ValueError(f"Unknown contrast: {contrast}")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return math.nan
    return 2.0 * int(np.logical_and(a, b).sum()) / denominator


def finite(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    return values_array[np.isfinite(values_array)]


def stats(values: list[float]) -> dict[str, float | int]:
    values_array = finite(values)
    if not len(values_array):
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
        }
    return {
        "n": int(len(values_array)),
        "mean": float(np.mean(values_array)),
        "sd": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else math.nan,
        "median": float(np.median(values_array)),
        "q1": float(np.percentile(values_array, 25)),
        "q3": float(np.percentile(values_array, 75)),
    }


def format_value(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def mean_sd(summary: dict[str, float | int]) -> str:
    return f"{format_value(float(summary['mean']))} ± {format_value(float(summary['sd']))}"


def median_iqr(summary: dict[str, float | int]) -> str:
    return (
        f"{format_value(float(summary['median']))} "
        f"[{format_value(float(summary['q1']))}–{format_value(float(summary['q3']))}]"
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(row.get(key), float) and math.isnan(row[key]) else row.get(key, "")
                for key in fieldnames
            })


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.cases.open(newline="") as stream:
        cases = list(csv.DictReader(stream, delimiter="\t"))

    expected = {f"{row['case_id']}.nii.gz" for row in cases}
    actual_gt = {path.name for path in args.gt_dir.glob("*.nii.gz")}
    actual_pred = {path.name for path in args.pred_dir.glob("*.nii.gz")}
    if expected != actual_gt or expected != actual_pred:
        raise RuntimeError(
            f"File mismatch: missing_gt={sorted(expected - actual_gt)}, "
            f"extra_gt={sorted(actual_gt - expected)}, "
            f"missing_pred={sorted(expected - actual_pred)}, "
            f"extra_pred={sorted(actual_pred - expected)}"
        )

    rows: list[dict] = []
    validation: list[str] = []
    allowed = {0, *LEVELS}
    for case in cases:
        filename = f"{case['case_id']}.nii.gz"
        gt_img = nib.load(args.gt_dir / filename)
        pred_img = nib.load(args.pred_dir / filename)
        if gt_img.shape != pred_img.shape:
            raise RuntimeError(f"Shape mismatch for {filename}: {gt_img.shape} != {pred_img.shape}")
        affine_error = float(np.max(np.abs(gt_img.affine - pred_img.affine)))
        if affine_error > 1e-4:
            raise RuntimeError(f"Affine mismatch for {filename}: max error {affine_error}")

        gt = np.rint(np.asanyarray(gt_img.dataobj)).astype(np.int16)
        pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int16)
        gt_labels = {int(value) for value in np.unique(gt)}
        pred_labels = {int(value) for value in np.unique(pred)}
        unexpected = (gt_labels | pred_labels) - allowed
        if unexpected:
            raise RuntimeError(f"Unexpected labels for {filename}: {sorted(unexpected)}")

        row: dict[str, object] = {
            "case_id": case["case_id"],
            "contrast": case["contrast"],
            "group": group_for_contrast(case["contrast"]),
            "reference": case["reference"],
            "binary_dice": dice(gt > 0, pred > 0),
            "gt_foreground_voxels": int((gt > 0).sum()),
            "pred_foreground_voxels": int((pred > 0).sum()),
            "gt_levels": ";".join(LEVELS[label] for label in LEVELS if label in gt_labels),
            "pred_levels": ";".join(LEVELS[label] for label in LEVELS if label in pred_labels),
            "max_affine_error": affine_error,
        }
        level_dice: list[float] = []
        for label, name in LEVELS.items():
            value = dice(gt == label, pred == label)
            row[f"dice_{name}"] = value
            if math.isfinite(value):
                level_dice.append(value)
        # First compute a 3D Dice for each level present in the union of the
        # prediction and reference, then macro-average those per-level values.
        row["macro_level_dice"] = float(np.mean(level_dice)) if level_dice else math.nan
        row["n_levels_in_union"] = len(level_dice)
        rows.append(row)
        validation.append(
            f"{case['case_id']}\tshape={gt.shape}\taffine_max_abs_diff={affine_error:.3g}"
            f"\tgt={sorted(gt_labels)}\tpred={sorted(pred_labels)}"
        )

    per_case_fields = [
        "case_id", "contrast", "group", "reference", "macro_level_dice", "binary_dice",
        *[f"dice_{name}" for name in LEVELS.values()],
        "n_levels_in_union", "gt_foreground_voxels", "pred_foreground_voxels",
        "gt_levels", "pred_levels", "max_affine_error",
    ]
    write_csv(args.out_dir / "metrics_per_case.csv", rows, per_case_fields)

    grouped_rows: list[dict] = []
    for group in ["All", "T2w", "INV1", "INV2", "UNIT1"]:
        selected = rows if group == "All" else [row for row in rows if row["group"] == group]
        macro = stats([float(row["macro_level_dice"]) for row in selected])
        binary = stats([float(row["binary_dice"]) for row in selected])
        grouped_rows.append({
            "group": group,
            "n_cases": len(selected),
            **{f"macro_{key}": value for key, value in macro.items()},
            **{f"binary_{key}": value for key, value in binary.items()},
        })
    group_fields = [
        "group", "n_cases", *[f"macro_{key}" for key in stats([])],
        *[f"binary_{key}" for key in stats([])],
    ]
    write_csv(args.out_dir / "metrics_by_group.csv", grouped_rows, group_fields)

    level_rows: list[dict] = []
    for label, name in LEVELS.items():
        values = [float(row[f"dice_{name}"]) for row in rows]
        summary = stats(values)
        level_rows.append({
            "label": label,
            "level": name,
            "n_valid": summary["n"],
            "n_gt_present": sum(name in str(row["gt_levels"]).split(";") for row in rows),
            "n_pred_present": sum(name in str(row["pred_levels"]).split(";") for row in rows),
            **{key: value for key, value in summary.items() if key != "n"},
        })
    level_fields = [
        "label", "level", "n_valid", "n_gt_present", "n_pred_present",
        "mean", "sd", "median", "q1", "q3",
    ]
    write_csv(args.out_dir / "metrics_by_level.csv", level_rows, level_fields)

    group_markdown = []
    for row in grouped_rows:
        macro = {key.removeprefix("macro_"): value for key, value in row.items() if key.startswith("macro_")}
        binary = {key.removeprefix("binary_"): value for key, value in row.items() if key.startswith("binary_")}
        group_markdown.append([
            str(row["group"]), str(row["n_cases"]), mean_sd(macro), median_iqr(macro),
            mean_sd(binary), median_iqr(binary),
        ])

    level_markdown = []
    for row in level_rows:
        summary = {key: row[key] for key in ["mean", "sd", "median", "q1", "q3"]}
        level_markdown.append([
            str(row["level"]), str(row["n_gt_present"]), str(row["n_pred_present"]),
            str(row["n_valid"]), mean_sd(summary), median_iqr(summary),
        ])

    case_markdown = [[
        str(row["case_id"]), str(row["group"]), format_value(float(row["macro_level_dice"])),
        format_value(float(row["binary_dice"])), str(row["gt_levels"]), str(row["pred_levels"]),
    ] for row in rows]

    text = "\n".join([
        "## Test Dice by contrast",
        "",
        "Macro level Dice is computed per image by first calculating a 3D Dice for each level present in the union of prediction and reference, then averaging those per-level Dice values. Values are mean ± sample SD and median [IQR]. Binary Dice collapses all rootlet levels to foreground.",
        "",
        markdown_table(
            ["Contrast", "n", "Macro level Dice", "Macro median [IQR]", "Binary Dice", "Binary median [IQR]"],
            group_markdown,
        ),
        "",
        "## Test Dice by spinal level",
        "",
        markdown_table(
            ["Level", "GT present", "Pred present", "n valid", "Dice", "Median [IQR]"],
            level_markdown,
        ),
        "",
        "## Per-image test Dice",
        "",
        markdown_table(
            ["Case", "Contrast", "Macro level Dice", "Binary Dice", "GT levels", "Predicted levels"],
            case_markdown,
        ),
        "",
        "Notes:",
        "",
        "- Dice is computed on the supplied prediction/reference grids with no post-processing.",
        "- A level absent from both prediction and reference is excluded (NA); a missed reference level scores 0.",
        "- The official `nnUNetv2_evaluate_folder` output may report a NaN aggregate when classes are absent from some images; these tables aggregate finite per-image values explicitly.",
    ])
    (args.out_dir / "dice_tables.md").write_text(text + "\n")
    (args.out_dir / "validation.txt").write_text("\n".join(validation) + "\n")
    print(text)


if __name__ == "__main__":
    main()
