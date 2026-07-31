#!/usr/bin/env python3
"""Plot per-class validation pseudo-Dice from a current nnU-Net training log.

Adapted from ivadomed/utilities/training_scripts/plot_nnunet_training_log.py
(Jan Valosek) to parse nnU-Net logs that render values as ``np.float32(...)``
and to support the ``all`` fold.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args()


def parse_float(token: str) -> float:
    token = token.strip()
    match = re.fullmatch(r"(?:np\.float32\()?([^()]+)\)?", token)
    if match is None:
        raise ValueError(f"Unrecognized pseudo-Dice value: {token!r}")
    return float(match.group(1))


def parse_log(path: Path) -> tuple[str, list[dict[str, object]]]:
    fold = "all"
    epoch: int | None = None
    rows: list[dict[str, object]] = []

    for line in path.read_text().splitlines():
        fold_match = re.search(r"Desired fold for training:\s+(\d+|all)", line)
        if fold_match:
            fold = fold_match.group(1)

        epoch_match = re.search(r"Epoch\s+(\d+)", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            continue

        dice_match = re.search(r"Pseudo dice \[(.*)\]", line)
        if dice_match and epoch is not None:
            rows.append(
                {
                    "epoch": epoch,
                    "pseudo_dice": [
                        parse_float(value)
                        for value in dice_match.group(1).split(",")
                    ],
                }
            )

    if not rows:
        raise ValueError(f"No completed epochs with pseudo-Dice found in {path}")
    return fold, rows


def main() -> None:
    args = parse_args()
    log_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else log_path.with_suffix(".png")
    )

    fold, rows = parse_log(log_path)
    width = len(rows[0]["pseudo_dice"])
    if any(len(row["pseudo_dice"]) != width for row in rows):
        raise ValueError(f"Inconsistent pseudo-Dice vector lengths in {log_path}")

    frame = pd.DataFrame(
        [
            {
                "epoch": row["epoch"],
                **{
                    f"validation_pseudo_dice_class_{index + 1}": value
                    for index, value in enumerate(row["pseudo_dice"])
                },
            }
            for row in rows
        ]
    )
    class_columns = list(frame.columns[1:])
    frame["validation_pseudo_dice_mean"] = frame[class_columns].mean(
        axis=1, skipna=True
    )

    figure = px.line(frame, x="epoch", y=frame.columns[1:])
    figure.update_traces(line={"width": 3})
    figure.update_traces(
        line={"color": "black", "width": 5},
        selector={"name": "validation_pseudo_dice_mean"},
    )
    figure.update_xaxes(title_text="Epoch")
    figure.update_yaxes(title_text="Validation Pseudo Dice", range=[-0.1, 1.1])
    figure.update_layout(
        title=f"Fold {fold} — Validation Pseudo Dice vs. Epoch",
        font={"size": 28},
        margin={"l": 80, "r": 50, "b": 50, "t": 100},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(output_path, width=1920, height=1080)
    print(f"Saved plot to {output_path}")
    print(f"Latest per-class pseudo-Dice: {frame.iloc[-1][class_columns].tolist()}")
    print(
        "Latest mean pseudo-Dice: "
        f"{frame.iloc[-1]['validation_pseudo_dice_mean']}"
    )


if __name__ == "__main__":
    main()
