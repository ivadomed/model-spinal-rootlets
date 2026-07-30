#!/usr/bin/env python3
"""Create nnU-Net ``splits_final.json`` from the reviewed rootlets CSV.

The cropped dataset uses descriptive nnU-Net case IDs, whereas the historical
CSV uses short IDs with numeric suffixes. This command maps between those
formats, omits rows marked Test from every training fold, and validates that no
available case is unassigned, duplicated across validation folds, or leaked
into training.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


# Legacy Dataset402 can omit these cases for documented historical reasons.
# A reviewed handoff dataset is allowed (and expected) to make them available.
KNOWN_UNAVAILABLE = {
    "sub-007_ses-headNormal_009",
    "sub-010_ses-headUp_015",
    "sub-amu02_215",
    "sub-barcelona01_212",
    "sub-brnoUhb01_085",
    "sub-brnoUhb03_209",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path,
                   help="Raw nnU-Net dataset containing imagesTr and dataset.json.")
    p.add_argument("--csv", required=True, type=Path,
                   help="Published MP2RAGE_T2w_fold_splits.csv.")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination splits_final.json in the preprocessed dataset directory.")
    return p


def csv_key(subject: str) -> str:
    """Remove the historical numeric sample index from T2w CSV identifiers."""
    return re.sub(r"_\d+$", "", subject)


def case_key(case_id: str) -> str:
    """Convert a cropped nnU-Net case ID to the corresponding CSV key."""
    if case_id.startswith("sub-"):
        # The reviewed handoff preserves the published CSV identifiers as its
        # nnU-Net case IDs. T2w identifiers carry a historical numeric suffix.
        return csv_key(case_id)
    if case_id.startswith("ds004507_"):
        key = case_id.removeprefix("ds004507_")
        if not key.endswith("_T2w"):
            raise ValueError(f"Unexpected ds004507 case ID: {case_id}")
        return key.removesuffix("_T2w")
    if case_id.startswith("data-multi-subject_"):
        key = case_id.removeprefix("data-multi-subject_")
        if not key.endswith("_T2w"):
            raise ValueError(f"Unexpected data-multi-subject case ID: {case_id}")
        return key.removesuffix("_T2w")
    if case_id.startswith("hc-leipzig-7t-mp2rage_sub-sspr"):
        return "sub-" + case_id.removeprefix("hc-leipzig-7t-mp2rage_sub-sspr")
    raise ValueError(f"Unrecognized cropped case ID: {case_id}")


def raw_case_ids(dataset: Path) -> set[str]:
    images = dataset / "imagesTr"
    if not images.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory: {images}")
    case_ids = {
        path.name.removesuffix("_0000.nii.gz")
        for path in images.glob("*_0000.nii.gz")
    }
    if not case_ids:
        raise ValueError(f"No training images found in {images}")
    return case_ids


def validate_cross_validation_coverage(
    splits: list[dict[str, list[str]]],
    expected_case_ids: set[str],
) -> None:
    """Require every non-test case to occur in validation exactly once."""
    validation_counts = {
        case_id: sum(case_id in split["val"] for split in splits)
        for case_id in expected_case_ids
    }
    missing = sorted(case_id for case_id, count in validation_counts.items() if count == 0)
    duplicated = sorted(
        case_id for case_id, count in validation_counts.items() if count > 1
    )
    unexpected = sorted(
        {
            case_id
            for split in splits
            for case_id in split["val"]
            if case_id not in expected_case_ids
        }
    )
    if missing or duplicated or unexpected:
        raise ValueError(
            "Validation folds must partition every non-test case exactly once; "
            f"missing={missing}, duplicated={duplicated}, unexpected={unexpected}"
        )


def build_splits(dataset: Path, csv_path: Path) -> tuple[list[dict[str, list[str]]], dict]:
    actual_ids = raw_case_ids(dataset)
    by_key: dict[str, str] = {}
    for case_id in actual_ids:
        key = case_key(case_id)
        if key in by_key:
            raise ValueError(f"Multiple cropped cases map to CSV key {key!r}")
        by_key[key] = case_id

    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fold_names = [f"Fold_{index}" for index in range(5)]
    required_columns = {"Subject", *fold_names}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

    mapped: dict[str, str] = {}
    missing: set[str] = set()
    for row in rows:
        subject = row["Subject"]
        case_id = by_key.get(csv_key(subject))
        if case_id is None:
            missing.add(subject)
        else:
            mapped[subject] = case_id

    test_subjects = {
        row["Subject"]
        for row in rows
        if {row[name].strip().lower() for name in fold_names} == {"test"}
    }
    inconsistent_test = {
        row["Subject"]
        for row in rows
        if "test" in {row[name].strip().lower() for name in fold_names}
        and row["Subject"] not in test_subjects
    }
    if inconsistent_test:
        raise ValueError(f"Test assignment differs across folds: {sorted(inconsistent_test)}")

    # A standard nnU-Net raw dataset can keep the held-out cohort in imagesTs
    # rather than imagesTr. Legacy Dataset402 also lacks six documented cases.
    allowed_missing = KNOWN_UNAVAILABLE | test_subjects
    unexpected_missing = missing - allowed_missing
    if unexpected_missing:
        raise ValueError(f"Unexpected CSV cases missing from dataset: {sorted(unexpected_missing)}")

    mapped_ids = set(mapped.values())
    unreferenced = actual_ids - mapped_ids
    if unreferenced:
        raise ValueError(f"Cropped cases absent from split CSV: {sorted(unreferenced)}")

    splits: list[dict[str, list[str]]] = []
    test_ids: set[str] = {
        case_id for subject, case_id in mapped.items() if subject in test_subjects
    }
    for fold_name in fold_names:
        train: list[str] = []
        val: list[str] = []
        for row in rows:
            case_id = mapped.get(row["Subject"])
            if case_id is None:
                continue
            assignment = row[fold_name].strip().lower()
            if assignment == "train":
                train.append(case_id)
            elif assignment == "validation":
                val.append(case_id)
            elif assignment == "test":
                test_ids.add(case_id)
            else:
                raise ValueError(
                    f"Invalid assignment {row[fold_name]!r} for {row['Subject']} in {fold_name}"
                )
        train, val = sorted(train), sorted(val)
        overlap = set(train) & set(val)
        if overlap:
            raise ValueError(f"Train/validation overlap in {fold_name}: {sorted(overlap)}")
        if not train or not val:
            raise ValueError(f"{fold_name} has an empty train or validation set")
        splits.append({"train": train, "val": val})

    non_test_ids = actual_ids - test_ids
    for index, split in enumerate(splits):
        assigned = set(split["train"]) | set(split["val"])
        if assigned != non_test_ids:
            raise ValueError(
                f"Fold_{index} does not cover every available non-test case; "
                f"missing={sorted(non_test_ids - assigned)}, extra={sorted(assigned - non_test_ids)}"
            )
        leaked = assigned & test_ids
        if leaked:
            raise ValueError(f"Test leakage in Fold_{index}: {sorted(leaked)}")
    validate_cross_validation_coverage(splits, non_test_ids)

    summary = {
        "available": len(actual_ids),
        "training_validation": len(non_test_ids),
        "test_excluded": len(test_subjects),
        "test_present_in_imagesTr": len(test_ids),
        "known_unavailable": sorted((missing & KNOWN_UNAVAILABLE) - test_subjects),
        "test_absent_from_imagesTr": sorted(missing & test_subjects),
    }
    return splits, summary


def main() -> None:
    args = parser().parse_args()
    splits, summary = build_splits(args.dataset.resolve(), args.csv.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits, indent=2) + "\n")
    print(f"Wrote {len(splits)} validated folds to {output}")
    print(
        f"Available={summary['available']}, "
        f"train/validation={summary['training_validation']}, "
        f"test excluded={summary['test_excluded']}"
    )
    print(f"Known unavailable cases: {', '.join(summary['known_unavailable'])}")


if __name__ == "__main__":
    main()
