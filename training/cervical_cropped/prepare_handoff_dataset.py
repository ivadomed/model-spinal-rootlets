#!/usr/bin/env python3
"""Build the cropped RPI dataset from the reviewed Rootlets label handoff.

The handoff contains labels only. This command uses the published split CSV to
pair its 76 training/validation and 17 test labels with the original images,
losslessly reconciles orientation-only grid differences, applies one fixed
image-derived crop policy, and writes a separate nnU-Net dataset.

No source image or handoff label is modified. Test labels are used only after
crop detection to verify that the fixed detector crop retained all foreground.
If any crop loses a labelled voxel, the command stops instead of expanding that
individual crop from its reference segmentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

try:
    from sc_crop import crop, detect
    from sc_crop.qc import check_label_crop
except ImportError as error:  # pragma: no cover - environment dependent
    raise SystemExit(
        "sc-crop is required. Run this command in the rootlets-romane environment "
        "or another environment where `import sc_crop` works."
    ) from error


FOLD_COLUMNS = tuple(f"Fold_{index}" for index in range(5))
TARGET_NIB_AXCODES = ("L", "A", "S")  # SCT's historical RPI convention.
TARGET_ORIENTATION = axcodes2ornt(TARGET_NIB_AXCODES)
GRID_ATOL = 2e-4
LABELS = {"background": 0, **{f"lvl{index}": index for index in range(1, 10)}}

# sc-crop 0.6.0 defaults plus at least 10 mm on every face. A full 93-case
# detector audit showed that anterior +10 mm still missed one label by 5.46 mm,
# so the fixed anterior margin is +25 mm. This replaces per-case expansion.
DEFAULT_PADDING_MM = {
    "superior": 50.0,
    "inferior": 110.0,
    "left": 25.0,
    "right": 25.0,
    "anterior": 40.0,
    "posterior": 32.0,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    label: Path
    image_candidates: tuple[Path, ...]


@dataclass(frozen=True)
class Pairing:
    case: Case
    image: Path
    alignment: str
    affine_delta_before: float
    orientation_transform: tuple[tuple[float, float], ...]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path,
                   help="Root containing ds004507, data-multi-subject, and hc-leipzig-7t-mp2rage.")
    p.add_argument("--labels-root", required=True, type=Path,
                   help="Immutable Rootlets-train-test-labels handoff.")
    p.add_argument("--csv", required=True, type=Path,
                   help="Published MP2RAGE_T2w_fold_splits.csv.")
    p.add_argument("--output", required=True, type=Path,
                   help="New derived nnU-Net dataset directory.")
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                   help="Device passed to sc-crop.")
    for face, default in DEFAULT_PADDING_MM.items():
        p.add_argument(f"--pad-{face}", type=float, default=default,
                       help=f"Fixed {face} padding in mm (default: {default:g}).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true",
                      help="Replace an existing derived output directory.")
    mode.add_argument("--resume", action="store_true",
                      help="Validate and reuse complete existing output pairs.")
    p.add_argument("--audit-only", action="store_true",
                   help="Audit inventory, label values, and source grids without detecting crops or writing.")
    p.add_argument("--detector-audit-only", action="store_true",
                   help="Also run detector crop QC for all cases, report required extra padding, and write nothing.")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_test_row(row: dict[str, str]) -> bool:
    assignments = {row[column].strip().lower() for column in FOLD_COLUMNS}
    if "test" in assignments and assignments != {"test"}:
        raise ValueError(f"{row['Subject']}: Test assignment is inconsistent across folds")
    return assignments == {"test"}


def source_candidates(data_root: Path, subject: str) -> tuple[Path, ...]:
    if "_ses-" in subject:
        acquisition = re.sub(r"_\d+$", "", subject)
        subject_id, session_id = acquisition.split("_", 2)[:2]
        image = (
            data_root / "ds004507" / subject_id / session_id / "anat"
            / f"{acquisition}_T2w.nii.gz"
        )
        return (image,)

    if re.search(r"_\d+$", subject):
        subject_id = re.sub(r"_\d+$", "", subject)
        filename = f"{subject_id}_T2w.nii.gz"
        return (
            data_root / "data-multi-subject" / subject_id / "anat" / filename,
            data_root / "data-multi-subject" / "sourcedata" / subject_id / "anat" / filename,
        )

    match = re.fullmatch(r"sub-(\d+)_(.+)", subject)
    if not match:
        raise ValueError(f"Cannot map CSV subject to an image: {subject}")
    subject_id = f"sub-sspr{match.group(1)}"
    filename = f"{subject_id}_{match.group(2)}.nii.gz"
    return (data_root / "hc-leipzig-7t-mp2rage" / subject_id / "anat" / filename,)


def discover_cases(data_root: Path, labels_root: Path, csv_path: Path) -> tuple[list[Case], dict]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"Subject", *FOLD_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Split CSV must contain {sorted(required)}")

    seen: set[str] = set()
    cases: list[Case] = []
    expected = {"labelsTr": set(), "labelsTs": set()}
    for row in rows:
        subject = row["Subject"].strip()
        if not subject or subject in seen:
            raise ValueError(f"Empty or duplicate split-CSV subject: {subject!r}")
        seen.add(subject)
        split = "labelsTs" if is_test_row(row) else "labelsTr"
        label = labels_root / split / f"{subject}.nii.gz"
        expected[split].add(label.name)
        cases.append(Case(subject, split, label, source_candidates(data_root, subject)))

    inventory: dict[str, dict[str, list[str] | int]] = {}
    for split in ("labelsTr", "labelsTs"):
        directory = labels_root / split
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        actual = {path.name for path in directory.glob("*.nii.gz")}
        missing = sorted(expected[split] - actual)
        if missing:
            raise ValueError(f"Missing expected {split} labels: {missing}")
        inventory[split] = {
            "expected": len(expected[split]),
            "actual": len(actual),
            "missing": missing,
            "extra": sorted(actual - expected[split]),
        }
    return cases, inventory


def orientation_transform(
    source: nib.spatialimages.SpatialImage,
    target: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    return ornt_transform(io_orientation(source.affine), io_orientation(target.affine))


def candidate_match(
    image: nib.spatialimages.SpatialImage,
    label: nib.spatialimages.SpatialImage,
) -> tuple[int, str, float, np.ndarray] | None:
    if image.shape[:3] == label.shape[:3] and np.allclose(
        image.affine, label.affine, atol=GRID_ATOL
    ):
        delta = float(np.max(np.abs(image.affine - label.affine)))
        transform = orientation_transform(label, image)
        return 0, "exact", delta, transform

    transform = orientation_transform(label, image)
    reoriented = label.as_reoriented(transform)
    if reoriented.shape[:3] == image.shape[:3] and np.allclose(
        reoriented.affine, image.affine, atol=GRID_ATOL
    ):
        delta = float(np.max(np.abs(reoriented.affine - image.affine)))
        return 1, "orientation_only", delta, transform
    return None


def choose_pairing(case: Case) -> Pairing:
    label = nib.load(case.label)
    matches: list[tuple[int, int, Path, str, float, np.ndarray]] = []
    present: list[str] = []
    for order, image_path in enumerate(case.image_candidates):
        if not image_path.is_file():
            continue
        present.append(str(image_path))
        match = candidate_match(nib.load(image_path), label)
        if match is not None:
            rank, alignment, delta, transform = match
            matches.append((rank, order, image_path, alignment, delta, transform))
    if not matches:
        raise ValueError(
            f"{case.case_id}: no source image has a grid equivalent to the handoff label. "
            f"Present candidates: {present or 'none'}"
        )
    _, _, image, alignment, delta, transform = min(matches, key=lambda item: item[:2])
    return Pairing(
        case=case,
        image=image,
        alignment=alignment,
        affine_delta_before=delta,
        orientation_transform=tuple(tuple(float(value) for value in row) for row in transform),
    )


def validate_label(label: nib.spatialimages.SpatialImage, case_id: str) -> tuple[list[int], int]:
    data = np.asanyarray(label.dataobj)
    values = np.unique(data)
    if not np.all(np.isfinite(values)) or not np.allclose(values, np.round(values)):
        raise ValueError(f"{case_id}: label contains non-finite or non-integer values: {values}")
    integer_values = [int(value) for value in values]
    unexpected = sorted(set(integer_values) - set(range(10)))
    if unexpected:
        raise ValueError(f"{case_id}: label values outside 0..9: {unexpected}")
    foreground = int(np.count_nonzero(data))
    if not foreground:
        raise ValueError(f"{case_id}: label is empty")
    return integer_values, foreground


def harmonize_label_to_image(
    label: nib.spatialimages.SpatialImage,
    image: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    transform = orientation_transform(label, image)
    aligned = label.as_reoriented(transform)
    if aligned.shape[:3] != image.shape[:3] or not np.allclose(
        aligned.affine, image.affine, atol=GRID_ATOL
    ):
        raise ValueError("Image and label are not equivalent under lossless axis permutation/flips")

    # The remaining sub-voxel differences are NIfTI header round-off. Start
    # from the image header so qform/sform codes and their stored coefficients
    # are byte-identical for ITK, which may consult a different form than
    # NiBabel's effective affine. Restore label-specific datatype/scaling.
    header = image.header.copy()
    header.set_data_dtype(aligned.get_data_dtype())
    header.set_slope_inter(1.0, 0.0)
    header["cal_min"] = 0
    header["cal_max"] = 9
    output = nib.Nifti1Image(np.asanyarray(aligned.dataobj), image.affine, header)
    return output


def reorient_rpi(image: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    transform = ornt_transform(io_orientation(image.affine), TARGET_ORIENTATION)
    return image.as_reoriented(transform)


def same_grid(
    image: nib.spatialimages.SpatialImage,
    label: nib.spatialimages.SpatialImage,
    atol: float = 1e-5,
) -> bool:
    return image.shape[:3] == label.shape[:3] and np.allclose(
        image.affine, label.affine, atol=atol
    )


def save_atomically(image: nib.spatialimages.SpatialImage, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.nii.gz")
    try:
        nib.save(image, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def output_paths(output: Path, pairing: Pairing) -> tuple[Path, Path]:
    suffix = "Tr" if pairing.case.split == "labelsTr" else "Ts"
    return (
        output / f"images{suffix}" / f"{pairing.case.case_id}_0000.nii.gz",
        output / f"labels{suffix}" / f"{pairing.case.case_id}.nii.gz",
    )


def prepare_output(output: Path, overwrite: bool, resume: bool) -> None:
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        elif not resume:
            raise FileExistsError(f"{output} exists; pass --resume or --overwrite")
    for directory in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (output / directory).mkdir(parents=True, exist_ok=True)


def padding_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        face: float(getattr(args, f"pad_{face}"))
        for face in DEFAULT_PADDING_MM
    }


def write_metadata(
    output: Path,
    rows: list[dict],
    inventory: dict,
    padding: dict[str, float],
    labels_root: Path,
    csv_path: Path,
) -> None:
    training_count = sum(row["split"] == "labelsTr" and row["status"] in {"processed", "reused"}
                         for row in rows)
    dataset = {
        "channel_names": {"0": "MRI"},
        "labels": LABELS,
        "numTraining": training_count,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    provenance = {
        "schema_version": 1,
        "source_labels_root": str(labels_root),
        "split_csv": str(csv_path),
        "crop_policy": "image-derived fixed padding; labels used only for post-detection QC",
        "padding_mm": padding,
        "inventory": inventory,
        "cases": rows,
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (output / "crop_qc.csv").open("w", newline="") as stream:
        fields = [
            "case_id", "split", "status", "alignment", "image", "label",
            "foreground_voxels", "has_label_9", "crop_retains_all_foreground",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def audit(pairings: list[Pairing], inventory: dict) -> dict:
    alignments: dict[str, int] = {}
    label_9_cases = 0
    without_label_9: list[str] = []
    for pairing in pairings:
        alignments[pairing.alignment] = alignments.get(pairing.alignment, 0) + 1
        values, _ = validate_label(nib.load(pairing.case.label), pairing.case.case_id)
        label_9_cases += 9 in values
        if 9 not in values:
            without_label_9.append(pairing.case.case_id)
    summary = {
        "cases": len(pairings),
        "training_validation": sum(pairing.case.split == "labelsTr" for pairing in pairings),
        "test": sum(pairing.case.split == "labelsTs" for pairing in pairings),
        "alignments": alignments,
        "cases_with_label_9": label_9_cases,
        "cases_without_label_9": without_label_9,
        "inventory": inventory,
    }
    return summary


def process_case(
    pairing: Pairing,
    output: Path,
    padding: dict[str, float],
    device: str | None,
) -> dict:
    image_path, label_path = output_paths(output, pairing)
    image = nib.load(pairing.image)
    source_label = nib.load(pairing.case.label)
    values, foreground = validate_label(source_label, pairing.case.case_id)
    label = harmonize_label_to_image(source_label, image)
    if not same_grid(image, label):
        raise RuntimeError(f"{pairing.case.case_id}: grid harmonization failed")

    bbox = detect(
        image,
        pad_superior=padding["superior"],
        pad_inferior=padding["inferior"],
        pad_left=padding["left"],
        pad_right=padding["right"],
        pad_anterior=padding["anterior"],
        pad_posterior=padding["posterior"],
        device=device,
    )
    qc = check_label_crop(label, bbox)
    if not qc["ok"]:
        raise RuntimeError(
            f"{pairing.case.case_id}: fixed detector crop loses label foreground; "
            f"required extra padding (mm): "
            f"{ {key: value for key, value in qc.items() if key.startswith('extra_pad_')} }"
        )

    cropped_image = crop(image, bbox)
    cropped_label = crop(label, bbox)
    if int(np.count_nonzero(np.asanyarray(cropped_label.dataobj))) != foreground:
        raise RuntimeError(f"{pairing.case.case_id}: crop QC and saved foreground disagree")

    rpi_image = reorient_rpi(cropped_image)
    rpi_label = reorient_rpi(cropped_label)
    rpi_label = harmonize_label_to_image(rpi_label, rpi_image)
    if not same_grid(rpi_image, rpi_label):
        raise RuntimeError(f"{pairing.case.case_id}: final RPI grids differ")
    save_atomically(rpi_image, image_path)
    save_atomically(rpi_label, label_path)

    bbox_fields = {
        key: int(bbox[key])
        for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    }
    return {
        "case_id": pairing.case.case_id,
        "split": pairing.case.split,
        "status": "processed",
        "image": str(pairing.image),
        "label": str(pairing.case.label),
        "label_sha256": sha256(pairing.case.label),
        "alignment": pairing.alignment,
        "orientation_transform": pairing.orientation_transform,
        "affine_delta_before_harmonization": pairing.affine_delta_before,
        "label_values": values,
        "foreground_voxels": foreground,
        "has_label_9": 9 in values,
        "detector_bbox": bbox_fields,
        "padding_mm": padding,
        "crop_qc": qc,
        "crop_retains_all_foreground": True,
        "output_shape": list(rpi_image.shape[:3]),
        "output_orientation_nibabel": "".join(nib.aff2axcodes(rpi_image.affine)),
    }


def detector_audit(
    pairings: list[Pairing],
    padding: dict[str, float],
    device: str | None,
) -> dict:
    maxima = {
        f"extra_pad_{face}_mm": 0.0
        for face in ("superior", "inferior", "left", "right", "anterior", "posterior")
    }
    failures: list[dict] = []
    for index, pairing in enumerate(pairings, start=1):
        print(f"[detector audit {index}/{len(pairings)}] {pairing.case.case_id}")
        image = nib.load(pairing.image)
        label = harmonize_label_to_image(nib.load(pairing.case.label), image)
        bbox = detect(
            image,
            pad_superior=padding["superior"],
            pad_inferior=padding["inferior"],
            pad_left=padding["left"],
            pad_right=padding["right"],
            pad_anterior=padding["anterior"],
            pad_posterior=padding["posterior"],
            device=device,
        )
        qc = check_label_crop(label, bbox)
        extra = {
            key: float(qc.get(key, 0.0))
            for key in maxima
        }
        for key, value in extra.items():
            maxima[key] = max(maxima[key], value)
        if not qc["ok"]:
            failures.append({"case_id": pairing.case.case_id, **extra})
    return {
        "padding_mm": padding,
        "failed_cases": len(failures),
        "max_extra_padding_mm": maxima,
        "failures": failures,
    }


def reused_row(pairing: Pairing, output: Path, previous: dict[str, dict]) -> dict:
    image_path, label_path = output_paths(output, pairing)
    if image_path.exists() != label_path.exists():
        raise RuntimeError(f"{pairing.case.case_id}: partial output pair exists")
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image, label = nib.load(image_path), nib.load(label_path)
    if not same_grid(image, label) or tuple(nib.aff2axcodes(image.affine)) != TARGET_NIB_AXCODES:
        raise RuntimeError(f"{pairing.case.case_id}: existing output pair failed grid/RPI validation")
    row = dict(previous.get(pairing.case.case_id, {}))
    if not row:
        values, foreground = validate_label(label, pairing.case.case_id)
        row = {
            "case_id": pairing.case.case_id,
            "split": pairing.case.split,
            "image": str(pairing.image),
            "label": str(pairing.case.label),
            "alignment": pairing.alignment,
            "foreground_voxels": foreground,
            "has_label_9": 9 in values,
            "crop_retains_all_foreground": None,
        }
    row["status"] = "reused"
    return row


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    labels_root = args.labels_root.resolve()
    csv_path = args.csv.resolve()
    for path in (data_root, labels_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    cases, inventory = discover_cases(data_root, labels_root, csv_path)
    pairings = [choose_pairing(case) for case in cases]
    summary = audit(pairings, inventory)
    print(json.dumps(summary, indent=2))
    if args.audit_only and args.detector_audit_only:
        raise ValueError("Choose only one of --audit-only and --detector-audit-only")
    if args.audit_only:
        print("Audit only: no crop detection or output writing was performed.")
        return
    padding = padding_from_args(args)
    if args.detector_audit_only:
        print(json.dumps(detector_audit(pairings, padding, args.device), indent=2))
        print("Detector audit only: no dataset output was written.")
        return

    output = args.output.resolve()
    prepare_output(output, args.overwrite, args.resume)
    manifest_path = output / "dataset_manifest.json"
    previous: dict[str, dict] = {}
    if args.resume and manifest_path.is_file():
        prior_document = json.loads(manifest_path.read_text())
        previous = {row["case_id"]: row for row in prior_document.get("cases", [])}

    rows: list[dict] = []
    for index, pairing in enumerate(pairings, start=1):
        image_path, label_path = output_paths(output, pairing)
        print(f"[{index}/{len(pairings)}] {pairing.case.case_id} ({pairing.case.split})")
        if image_path.exists() or label_path.exists():
            if not args.resume:
                raise FileExistsError(f"Output exists for {pairing.case.case_id}")
            row = reused_row(pairing, output, previous)
        else:
            row = process_case(pairing, output, padding, args.device)
        rows.append(row)
        write_metadata(output, rows, inventory, padding, labels_root, csv_path)

    training = sum(row["split"] == "labelsTr" for row in rows)
    testing = sum(row["split"] == "labelsTs" for row in rows)
    retained = sum(row.get("crop_retains_all_foreground") is True for row in rows)
    if training != 76 or testing != 17:
        raise RuntimeError(f"Expected 76 training/validation and 17 test cases, got {training} and {testing}")
    print(
        f"Wrote {training} training/validation and {testing} test pairs to {output}; "
        f"{retained} newly processed crops retained all reference foreground."
    )


def main() -> None:
    try:
        run(parser().parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
