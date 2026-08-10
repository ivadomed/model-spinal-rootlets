#!/usr/bin/env python3
"""Render comparable nnU-Net fallback training curves from plain-text logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PATTERNS = {
    "epoch": re.compile(r"Epoch (\d+)\s*$"),
    "learning_rate": re.compile(r"Current learning rate: ([-+0-9.eE]+)"),
    "train_loss": re.compile(r"train_loss ([-+0-9.eE]+)"),
    "val_loss": re.compile(r"val_loss ([-+0-9.eE]+)"),
    "pseudo_dice": re.compile(
        r"Pseudo dice \[np\.float32\(([-+0-9.eE]+)\), np\.float32\(([-+0-9.eE]+)\)\]"
    ),
    "epoch_time": re.compile(r"Epoch time: ([-+0-9.eE]+) s"),
}


def parse_log(path: Path) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    epoch: int | None = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        match = PATTERNS["epoch"].search(line)
        if match:
            epoch = int(match.group(1))
            records.setdefault(epoch, {"epoch": epoch})
            continue
        if epoch is None:
            continue
        for field in ("learning_rate", "train_loss", "val_loss", "epoch_time"):
            match = PATTERNS[field].search(line)
            if match:
                records[epoch][field] = float(match.group(1))
        match = PATTERNS["pseudo_dice"].search(line)
        if match:
            records[epoch]["dorsal_pseudo_dice"] = float(match.group(1))
            records[epoch]["ventral_pseudo_dice"] = float(match.group(2))
    return [records[index] for index in sorted(records) if "val_loss" in records[index]]


def render(candidates: list[tuple[str, Path]], output_png: Path, output_csv: Path) -> None:
    parsed = {name: parse_log(path) for name, path in candidates}
    if any(not records for records in parsed.values()):
        raise ValueError("Every candidate log must contain at least one complete epoch.")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for name, records in parsed.items():
        epoch = [record["epoch"] for record in records]
        axes[0, 0].plot(epoch, [record["train_loss"] for record in records], label=f"{name} train")
        axes[0, 0].plot(epoch, [record["val_loss"] for record in records], linestyle="--", label=f"{name} val")
        axes[0, 1].plot(epoch, [record["dorsal_pseudo_dice"] for record in records], label=f"{name} dorsal")
        axes[0, 1].plot(epoch, [record["ventral_pseudo_dice"] for record in records], linestyle="--", label=f"{name} ventral")
        axes[1, 0].plot(epoch, [record["learning_rate"] for record in records], label=name)
        axes[1, 1].plot(epoch, [record["epoch_time"] for record in records], label=name)
    settings = (
        (axes[0, 0], "Loss", "Loss"),
        (axes[0, 1], "Pseudo-Dice", "Score"),
        (axes[1, 0], "Learning-rate schedule", "Learning rate"),
        (axes[1, 1], "Epoch time", "Seconds"),
    )
    for axis, title, ylabel in settings:
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Fallback architecture training · 10 train / 2 validation cases\n"
        "Pseudo-Dice is an nnU-Net training diagnostic, not held-out test accuracy",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    fields = (
        "candidate",
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "dorsal_pseudo_dice",
        "ventral_pseudo_dice",
        "epoch_time",
    )
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, records in parsed.items():
            for record in records:
                writer.writerow({"candidate": name, **record})


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Candidate must use NAME=TRAINING_LOG.")
    return name, Path(path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, type=_candidate)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    render(args.candidate, args.output_png, args.output_csv)
    print(args.output_png)
    print(args.output_csv)


if __name__ == "__main__":
    main()
