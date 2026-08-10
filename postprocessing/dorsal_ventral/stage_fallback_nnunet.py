#!/usr/bin/env python3
"""Stage a development-only nnU-Net dataset for the V5 fallback model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def stage_dataset(
    frozen_dataset: Path,
    output_dataset: Path,
) -> dict[str, Any]:
    manifest = json.loads((frozen_dataset / "hybrid_v5_split.json").read_text())
    split = manifest["split"]
    development_cases = split["train"] + split["validation"]
    test_cases = split["test"]

    directories = {
        "imagesTr": output_dataset / "imagesTr",
        "labelsTr": output_dataset / "labelsTr",
        "imagesTs": output_dataset / "imagesTs",
        "labelsTs": output_dataset / "labelsTs",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for case_id in development_cases:
        for channel in ("0000", "0001"):
            shutil.copy2(
                frozen_dataset / "imagesTr" / f"{case_id}_{channel}.nii.gz",
                directories["imagesTr"] / f"{case_id}_{channel}.nii.gz",
            )
        shutil.copy2(
            frozen_dataset / "labelsTr" / f"{case_id}.nii.gz",
            directories["labelsTr"] / f"{case_id}.nii.gz",
        )

    for case_id in test_cases:
        for channel in ("0000", "0001"):
            shutil.copy2(
                frozen_dataset / "imagesTr" / f"{case_id}_{channel}.nii.gz",
                directories["imagesTs"] / f"{case_id}_{channel}.nii.gz",
            )
        shutil.copy2(
            frozen_dataset / "labelsTr" / f"{case_id}.nii.gz",
            directories["labelsTs"] / f"{case_id}.nii.gz",
        )

    dataset_json = {
        "channel_names": {"0": "MRI", "1": "level-labelled rootlet support"},
        "labels": {"background": 0, "dorsal": 1, "ventral": 2},
        "numTraining": len(development_cases),
        "file_ending": ".nii.gz",
    }
    (output_dataset / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n"
    )
    splits_final = [
        {
            "train": split["train"],
            "val": split["validation"],
        }
    ]
    (output_dataset / "splits_final.json").write_text(
        json.dumps(splits_final, indent=2) + "\n"
    )
    staged = {
        "schema": "rootlet-dv-v5-fallback-nnunet-stage-v1",
        "source": str(frozen_dataset.resolve()),
        "dataset": str(output_dataset.resolve()),
        "development_cases": len(development_cases),
        "train_cases": len(split["train"]),
        "validation_cases": len(split["validation"]),
        "test_cases": len(test_cases),
        "split_seed": manifest["split_seed"],
        "split": split,
        "support_contract": (
            "At inference, use network D/V predictions only inside V5 fallback "
            "levels and clip them to the fixed RootletSeg support."
        ),
    }
    (output_dataset / "fallback_manifest.json").write_text(
        json.dumps(staged, indent=2) + "\n"
    )
    return staged


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    staged = stage_dataset(args.frozen_dataset, args.output_dataset)
    print(json.dumps({key: staged[key] for key in (
        "development_cases", "train_cases", "validation_cases", "test_cases"
    )}, indent=2))


if __name__ == "__main__":
    main()
