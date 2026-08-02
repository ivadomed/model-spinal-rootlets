#!/usr/bin/env python3
"""Measure paired-session stability without claiming anatomical accuracy.

The scans are not registered to one another, so this evaluator deliberately
avoids voxelwise overlap metrics. It compares per-level dorsal fractions,
predicted support volumes, label presence, and splitter fallback rates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np


MANIFEST_FIELDS = ("subject", "session", "combined", "dorsal", "ventral", "qc")
SCAN_FIELDS = (
    "subject",
    "session",
    "level",
    "combined_voxels",
    "combined_volume_mm3",
    "dorsal_voxels",
    "ventral_voxels",
    "dorsal_fraction",
    "components",
    "fallback_components",
    "fallback_fraction",
    "attachment_islands",
    "neutral_attachment_islands",
)
PAIR_FIELDS = (
    "subject",
    "session_a",
    "session_b",
    "level",
    "present_a",
    "present_b",
    "presence_agreement",
    "dorsal_fraction_a",
    "dorsal_fraction_b",
    "abs_dorsal_fraction_difference",
    "combined_volume_mm3_a",
    "combined_volume_mm3_b",
    "support_volume_relative_difference",
    "fallback_fraction_a",
    "fallback_fraction_b",
    "abs_fallback_fraction_difference",
)


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _load_integer_image(path: Path, name: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must be a finite 3D image: {path}")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-4) or np.any(rounded < 0):
        raise ValueError(f"{name} must contain non-negative integer labels: {path}")
    return image, rounded.astype(np.int32)


def _resolve_path(value: str, manifest_directory: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_directory / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Manifest has no scan rows.")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["subject"], row["session"])
        if not all(key):
            raise ValueError("Every manifest row needs non-empty subject and session values.")
        if key in seen:
            raise ValueError(f"Duplicate subject/session row: {key[0]} {key[1]}")
        seen.add(key)
        for field in ("combined", "dorsal", "ventral", "qc"):
            row[field] = str(_resolve_path(row[field], path.parent).resolve())
    return rows


def scan_metrics(row: dict[str, str]) -> list[dict[str, Any]]:
    combined_image, combined = _load_integer_image(Path(row["combined"]), "combined")
    dorsal_image, dorsal = _load_integer_image(Path(row["dorsal"]), "dorsal")
    ventral_image, ventral = _load_integer_image(Path(row["ventral"]), "ventral")
    for name, image, data in (
        ("dorsal", dorsal_image, dorsal),
        ("ventral", ventral_image, ventral),
    ):
        if image.shape != combined_image.shape or not np.allclose(
            image.affine, combined_image.affine, atol=1e-4
        ):
            raise ValueError(f"{name} grid differs from combined for {row['subject']} {row['session']}")
        if data.shape != combined.shape:
            raise ValueError(f"{name} shape differs from combined.")
    if np.any((dorsal > 0) & (ventral > 0)):
        raise ValueError("Dorsal and ventral outputs overlap.")
    if not np.array_equal(dorsal + ventral, combined):
        raise ValueError("Dorsal and ventral outputs do not exactly partition combined.")

    qc = json.loads(Path(row["qc"]).read_text())
    qc_levels = {int(item["label"]): item for item in qc.get("levels", [])}
    voxel_volume = float(abs(np.linalg.det(combined_image.affine[:3, :3])))
    levels = [int(value) for value in np.unique(combined[combined > 0])]
    records: list[dict[str, Any]] = []
    for level in ["all", *levels]:
        mask = combined > 0 if level == "all" else combined == level
        dorsal_mask = dorsal > 0 if level == "all" else dorsal == level
        ventral_mask = ventral > 0 if level == "all" else ventral == level
        combined_voxels = int(np.count_nonzero(mask))
        dorsal_voxels = int(np.count_nonzero(dorsal_mask & mask))
        ventral_voxels = int(np.count_nonzero(ventral_mask & mask))
        if level == "all":
            components = int(sum(int(item.get("components", 0)) for item in qc_levels.values()))
            fallbacks = int(qc.get("fallback_components", 0))
            attachments = int(
                sum(int(item.get("attachment_islands", 0)) for item in qc_levels.values())
            )
            neutral = int(
                sum(
                    int(item.get("neutral_attachment_islands", 0))
                    for item in qc_levels.values()
                )
            )
        else:
            level_qc = qc_levels.get(int(level), {})
            components = int(level_qc.get("components", 0))
            fallbacks = int(level_qc.get("fallback_components", 0))
            attachments = int(level_qc.get("attachment_islands", 0))
            neutral = int(level_qc.get("neutral_attachment_islands", 0))
        records.append(
            {
                "subject": row["subject"],
                "session": row["session"],
                "level": str(level),
                "combined_voxels": combined_voxels,
                "combined_volume_mm3": combined_voxels * voxel_volume,
                "dorsal_voxels": dorsal_voxels,
                "ventral_voxels": ventral_voxels,
                "dorsal_fraction": _optional_ratio(dorsal_voxels, combined_voxels),
                "components": components,
                "fallback_components": fallbacks,
                "fallback_fraction": _optional_ratio(fallbacks, components),
                "attachment_islands": attachments,
                "neutral_attachment_islands": neutral,
            }
        )
    return records


def pair_metrics(scan_records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    scans: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in scan_records:
        scans[(record["subject"], record["session"])][record["level"]] = record

    by_subject: dict[str, list[str]] = defaultdict(list)
    for subject, session in scans:
        by_subject[subject].append(session)

    records: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for subject, sessions in sorted(by_subject.items()):
        sessions = sorted(set(sessions))
        if len(sessions) != 2:
            incomplete.append(subject)
            continue
        session_a, session_b = sessions
        levels_a = scans[(subject, session_a)]
        levels_b = scans[(subject, session_b)]
        level_order = sorted(
            set(levels_a) | set(levels_b),
            key=lambda value: (-1 if value == "all" else int(value)),
        )
        for level in level_order:
            a = levels_a.get(level)
            b = levels_b.get(level)
            present_a = a is not None
            present_b = b is not None

            def value(record: dict[str, Any] | None, field: str) -> Any:
                return record[field] if record is not None else None

            dorsal_difference = None
            volume_difference = None
            fallback_difference = None
            if a is not None and b is not None:
                dorsal_difference = abs(a["dorsal_fraction"] - b["dorsal_fraction"])
                mean_volume = (a["combined_volume_mm3"] + b["combined_volume_mm3"]) / 2
                volume_difference = _optional_ratio(
                    abs(a["combined_volume_mm3"] - b["combined_volume_mm3"]), mean_volume
                )
                if a["fallback_fraction"] is not None and b["fallback_fraction"] is not None:
                    fallback_difference = abs(a["fallback_fraction"] - b["fallback_fraction"])
            records.append(
                {
                    "subject": subject,
                    "session_a": session_a,
                    "session_b": session_b,
                    "level": level,
                    "present_a": present_a,
                    "present_b": present_b,
                    "presence_agreement": present_a == present_b,
                    "dorsal_fraction_a": value(a, "dorsal_fraction"),
                    "dorsal_fraction_b": value(b, "dorsal_fraction"),
                    "abs_dorsal_fraction_difference": dorsal_difference,
                    "combined_volume_mm3_a": value(a, "combined_volume_mm3"),
                    "combined_volume_mm3_b": value(b, "combined_volume_mm3"),
                    "support_volume_relative_difference": volume_difference,
                    "fallback_fraction_a": value(a, "fallback_fraction"),
                    "fallback_fraction_b": value(b, "fallback_fraction"),
                    "abs_fallback_fraction_difference": fallback_difference,
                }
            )
    return records, incomplete


def _distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if not finite.size:
        return {"count": 0, "median": None, "iqr": None, "p95": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
        "p95": float(np.percentile(finite, 95)),
    }


def _highest(
    records: list[dict[str, Any]], field: str, limit: int = 10
) -> list[dict[str, Any]]:
    ranked = sorted(
        (record for record in records if record[field] is not None),
        key=lambda record: float(record[field]),
        reverse=True,
    )
    return [
        {
            "subject": record["subject"],
            "level": record["level"],
            "session_a": record["session_a"],
            "session_b": record["session_b"],
            field: record[field],
        }
        for record in ranked[:limit]
    ]


def summarize(pair_records: list[dict[str, Any]], incomplete: list[str]) -> dict[str, Any]:
    per_level = [record for record in pair_records if record["level"] != "all"]
    return {
        "interpretation": (
            "Unregistered paired-session robustness only; these values are not dorsal/ventral "
            "accuracy and do not substitute for expert labels."
        ),
        "complete_subject_pairs": len({record["subject"] for record in pair_records}),
        "incomplete_subjects": incomplete,
        "per_level_pair_rows": len(per_level),
        "level_presence_agreement_fraction": _optional_ratio(
            sum(bool(record["presence_agreement"]) for record in per_level), len(per_level)
        ),
        "abs_dorsal_fraction_difference": _distribution(
            record["abs_dorsal_fraction_difference"] for record in per_level
        ),
        "support_volume_relative_difference": _distribution(
            record["support_volume_relative_difference"] for record in per_level
        ),
        "abs_fallback_fraction_difference": _distribution(
            record["abs_fallback_fraction_difference"] for record in per_level
        ),
        "highest_abs_dorsal_fraction_differences": _highest(
            per_level, "abs_dorsal_fraction_difference"
        ),
        "highest_support_volume_relative_differences": _highest(
            per_level, "support_volume_relative_difference"
        ),
        "highest_abs_fallback_fraction_differences": _highest(
            per_level, "abs_fallback_fraction_difference"
        ),
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    rows = read_manifest(manifest)
    scan_records = [record for row in rows for record in scan_metrics(row)]
    paired_records, incomplete = pair_metrics(scan_records)
    summary = summarize(paired_records, incomplete)
    _write_csv(Path(args.output_scan_csv), SCAN_FIELDS, scan_records)
    _write_csv(Path(args.output_pair_csv), PAIR_FIELDS, paired_records)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-scan-csv", required=True)
    parser.add_argument("--output-pair-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    summary = run(get_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
