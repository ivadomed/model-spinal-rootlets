#!/usr/bin/env python3
"""Extract SCT spinal-cord and rootlet inference times from batch logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
COMMAND = re.compile(r"\bsct_deepseg\s+(spinalcord|rootlets)\b")
RUNTIME = re.compile(r"Total runtime;\s*([0-9]+(?:\.[0-9]+)?)\s*seconds")
FIELDS = (
    "subject",
    "session",
    "compute",
    "spinalcord_seconds",
    "rootlets_seconds",
    "total_sct_seconds",
    "post_rootlets_to_split_qc_seconds",
    "log",
)


def parse_sct_times(log_path: Path) -> dict[str, float]:
    """Return the last completed runtime for each SCT task in one log."""

    current_task: str | None = None
    timings: dict[str, float] = {}
    text = ANSI_ESCAPE.sub("", log_path.read_text(errors="replace"))
    for line in text.splitlines():
        command = COMMAND.search(line)
        if command:
            current_task = command.group(1)
            continue
        runtime = RUNTIME.search(line)
        if runtime and current_task is not None:
            timings[current_task] = float(runtime.group(1))
            current_task = None
    return timings


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
        return {"count": 0, "median": None, "iqr": None, "p95": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject", "session", "compute", "combined", "qc", "log"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing runtime columns: {', '.join(sorted(missing))}")
        manifest_rows = list(reader)
    if not manifest_rows:
        raise ValueError("Manifest has no scan rows.")

    rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for manifest_row in manifest_rows:
        log = Path(manifest_row["log"])
        if not log.is_absolute():
            log = manifest.parent / log
        timings = parse_sct_times(log)
        missing_tasks = sorted({"spinalcord", "rootlets"} - set(timings))
        if missing_tasks:
            incomplete.append(
                f"{manifest_row['subject']} {manifest_row['session']}: {','.join(missing_tasks)}"
            )
            continue
        combined = Path(manifest_row["combined"])
        qc = Path(manifest_row["qc"])
        if not combined.is_absolute():
            combined = manifest.parent / combined
        if not qc.is_absolute():
            qc = manifest.parent / qc
        post_rootlets_interval = qc.stat().st_mtime - combined.stat().st_mtime
        if post_rootlets_interval < 0:
            raise ValueError(
                f"Split QC predates RootletSeg output: {manifest_row['subject']} "
                f"{manifest_row['session']}"
            )
        rows.append(
            {
                "subject": manifest_row["subject"],
                "session": manifest_row["session"],
                "compute": manifest_row["compute"],
                "spinalcord_seconds": timings["spinalcord"],
                "rootlets_seconds": timings["rootlets"],
                "total_sct_seconds": timings["spinalcord"] + timings["rootlets"],
                "post_rootlets_to_split_qc_seconds": post_rootlets_interval,
                "log": str(log.resolve()),
            }
        )
    summary = {
        "scans_in_manifest": len(manifest_rows),
        "scans_with_complete_timings": len(rows),
        "compute_modes": sorted({row["compute"] for row in rows}),
        "spinalcord_seconds": _distribution(row["spinalcord_seconds"] for row in rows),
        "rootlets_seconds": _distribution(row["rootlets_seconds"] for row in rows),
        "total_sct_seconds": _distribution(row["total_sct_seconds"] for row in rows),
        "post_rootlets_to_split_qc_seconds": _distribution(
            row["post_rootlets_to_split_qc_seconds"] for row in rows
        ),
        "incomplete": incomplete,
        "scope": (
            "SCT-reported command runtime plus an approximate filesystem interval from the "
            "RootletSeg output write to split QC write; that interval includes SCT return, "
            "Python launch, deterministic split, and output writes"
        ),
    }
    return rows, summary


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], output_csv: Path, output_json: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    rows, summary = summarize(Path(args.manifest))
    write_outputs(rows, summary, Path(args.output_csv), Path(args.output_json))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
