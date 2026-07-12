#!/usr/bin/env python3
"""Build a cropped, single-channel nnU-Net v2 rootlet dataset from BIDS data.

The command discovers ``*_label-rootlets_dseg.nii.gz`` labels below each BIDS
root's ``derivatives/labels`` folder. For every labelled acquisition it finds
the supported anatomical images (T2w, UNIT1, INV1 and INV2) for that subject,
detects a spinal-cord crop with :mod:`sc_crop`, and applies one final bounding
box to *both* image and label.

The detector bbox is never allowed to discard a labelled rootlet: if necessary
it is enlarged to the label extent plus ``--label-padding-mm``. This is a
training-only safety net; it must not be used to hide systematic detector
errors, which are recorded in ``dataset_manifest.json``.

Example
-------
python training/cervical_cropped/prepare_dataset.py \\
  --input-root /data/data-multi-subject /data/ds004507 /data/hc-leipzig-7t-mp2rage \\
  --output /scratch/$USER/nnUNet_raw/Dataset401_CervicalRootletsCropped \\
  --device cuda


* Note *
Made specifically to serve these following labeled datasets:
- open-access ds004507: https://openneuro.org/datasets/ds004507/versions/1.1.1
- open-access spine-generic/data-multi-subject: https://github.com/spine-generic/data-multi-subject/tree/r20250314
- private MP2RAGE dataset (data.neuro.polymtl.ca/hc-leipzig-7t-mp2rage)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from sc_crop import crop, detect
    from sc_crop.qc import check_label_crop
except ImportError as error:  # pragma: no cover - depends on caller environment
    raise SystemExit(
        "sc-crop is required. Install it (for example: `pip install -e sc-crop`) "
        "or run this command from an environment where `import sc_crop` works."
    ) from error


LABEL_SUFFIX = "_label-rootlets_dseg.nii.gz"
SUPPORTED_SUFFIXES = (
    "_T2w.nii.gz",
    "_UNIT1.nii.gz",
    "_inv-1_part-mag_MP2RAGE.nii.gz",
    "_inv-2_part-mag_MP2RAGE.nii.gz",
)
LABELS = {"background": 0, **{f"lvl{i}": i for i in range(1, 10)}}


@dataclass(frozen=True)
class Case:
    source: str
    image: Path
    label: Path
    case_id: str


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True, nargs="+", type=Path,
                   help="One or more BIDS roots. Labels must be in derivatives/labels.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output nnU-Net dataset directory, e.g. .../nnUNet_raw/Dataset401_CervicalRootletsCropped.")
    p.add_argument("--label-padding-mm", type=float, default=5.0,
                   help="Extra padding on a face only when the detector crop misses a label (default: 5).")
    p.add_argument("--pad-rl", type=float, default=None, help="Override sc-crop's symmetric right-left padding (mm).")
    p.add_argument("--pad-ap", type=float, default=None, help="Override sc-crop's symmetric anterior-posterior padding (mm).")
    p.add_argument("--pad-si", type=float, default=None, help="Override sc-crop's symmetric superior-inferior padding (mm).")
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                   help="Device passed to sc-crop (default: sc-crop's own default).")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset directory.")
    p.add_argument("--dry-run", action="store_true", help="Discover and validate cases but do not run detection or write files.")
    return p


def source_name(root: Path) -> str:
    """Stable dataset prefix that prevents cross-dataset subject-ID collisions."""
    return re.sub(r"[^A-Za-z0-9]+", "-", root.name).strip("-")


def subject_id(case_id: str) -> str:
    """Return the BIDS subject token for grouping folds, if present."""
    match = re.search(r"(?:^|_)(sub-[^_]+)", case_id)
    if not match:
        raise ValueError(f"Cannot extract a BIDS subject ID from {case_id}")
    return match.group(1)


def discover_cases(root: Path) -> tuple[list[Case], list[str]]:
    """Pair each primary rootlet label with every supported image in its BIDS anat dir."""
    labels_root = root / "derivatives" / "labels"
    if not labels_root.is_dir():
        return [], [f"{root}: missing derivatives/labels"]

    cases: list[Case] = []
    skipped: list[str] = []
    for label in sorted(labels_root.rglob(f"*{LABEL_SUFFIX}")):
        if ".git" in label.parts or "_desc-" in label.name:
            # Rater/STAPLE derivatives are not an independent training sample.
            continue
        relative = label.relative_to(labels_root)
        image_dir = root / relative.parent
        if not image_dir.is_dir():
            skipped.append(f"{label}: matching image directory does not exist")
            continue
        subject_prefix = label.name.removesuffix(LABEL_SUFFIX).rsplit("_", 1)[0]
        images = [
            image_dir / f"{subject_prefix}{suffix}"
            for suffix in SUPPORTED_SUFFIXES
            if (image_dir / f"{subject_prefix}{suffix}").is_file()
        ]
        if not images:
            skipped.append(f"{label}: no supported image found")
            continue
        for image in images:
            contrast = image.name.removesuffix(".nii.gz").removeprefix(f"{subject_prefix}_")
            case_id = f"{source_name(root)}_{subject_prefix}_{contrast}"
            cases.append(Case(source_name(root), image, label, case_id))
    return cases, skipped


def assert_same_grid(image: nib.Nifti1Image, label: nib.Nifti1Image, case: Case) -> None:
    if image.shape[:3] != label.shape[:3] or not np.allclose(image.affine, label.affine, atol=1e-4):
        raise ValueError(
            f"{case.case_id}: image and label are not on the same voxel grid. "
            "Register/resample the label to the image before cropping."
        )


def expanded_bbox(bbox: dict, label: nib.Nifti1Image, padding_mm: float) -> tuple[dict, bool]:
    """Union detector bbox with labelled extent + padding, in native voxel space."""
    result = dict(bbox)
    data = np.asarray(label.dataobj)
    nonzero = np.argwhere(data != 0)
    if not len(nonzero):
        raise ValueError("Label is empty; refusing to create a training case.")
    shape = data.shape[:3]
    pad_vox = np.ceil(padding_mm / np.asarray(label.header.get_zooms()[:3])).astype(int)
    label_min, label_max = nonzero.min(axis=0), nonzero.max(axis=0)
    low_names, high_names = ("xmin", "ymin", "zmin"), ("xmax", "ymax", "zmax")
    changed = False
    for axis, (low_name, high_name) in enumerate(zip(low_names, high_names)):
        low = min(int(result[low_name]), max(0, int(label_min[axis] - pad_vox[axis])))
        high = max(int(result[high_name]), min(shape[axis] - 1, int(label_max[axis] + pad_vox[axis])))
        changed |= low != result[low_name] or high != result[high_name]
        result[low_name], result[high_name] = low, high
    return result, changed


def bbox_fields(bbox: dict) -> dict[str, int]:
    return {key: int(bbox[key]) for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}


def write_dataset_json(output: Path, count: int) -> None:
    dataset = {
        "channel_names": {"0": "MRI"},
        "labels": LABELS,
        "numTraining": count,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output)
    (output / "imagesTr").mkdir(parents=True)
    (output / "labelsTr").mkdir()


def build(args: argparse.Namespace) -> None:
    roots = [root.resolve() for root in args.input_root]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    all_cases: list[Case] = []
    skipped: list[str] = []
    for root in roots:
        cases, messages = discover_cases(root)
        all_cases.extend(cases)
        skipped.extend(messages)
    duplicate_ids = {case.case_id for case in all_cases if sum(c.case_id == case.case_id for c in all_cases) > 1}
    if duplicate_ids:
        raise ValueError(f"Duplicate nnU-Net case IDs: {sorted(duplicate_ids)}")
    if not all_cases:
        raise RuntimeError("No primary rootlet image/label pairs were discovered.")
    print(f"Discovered {len(all_cases)} image/label pairs from {len(roots)} source datasets.")
    if args.dry_run:
        print("Dry run: no detector inference or output was performed.")
        return

    output = args.output.resolve()
    prepare_output(output, args.overwrite)
    manifest: list[dict] = []
    for index, case in enumerate(all_cases, start=1):
        print(f"[{index}/{len(all_cases)}] {case.case_id}")
        image, label = nib.load(case.image), nib.load(case.label)
        assert_same_grid(image, label, case)
        detector_bbox = detect(case.image, pad_rl=args.pad_rl, pad_ap=args.pad_ap,
                               pad_si=args.pad_si, device=args.device)
        qc = check_label_crop(label, detector_bbox)
        final_bbox, expanded = expanded_bbox(detector_bbox, label, args.label_padding_mm)
        cropped_image, cropped_label = crop(image, final_bbox), crop(label, final_bbox)
        if np.count_nonzero(np.asarray(cropped_label.dataobj)) != np.count_nonzero(np.asarray(label.dataobj)):
            raise RuntimeError(f"{case.case_id}: final crop still loses label voxels")
        nib.save(cropped_image, output / "imagesTr" / f"{case.case_id}_0000.nii.gz")
        nib.save(cropped_label, output / "labelsTr" / f"{case.case_id}.nii.gz")
        manifest.append({
            "case_id": case.case_id, "source": case.source,
            "subject_id": subject_id(case.case_id),
            "split_group": f"{case.source}:{subject_id(case.case_id)}",
            "image": str(case.image), "label": str(case.label),
            "detector_bbox": bbox_fields(detector_bbox), "final_bbox": bbox_fields(final_bbox),
            "expanded_for_label": expanded, "detector_label_qc": qc,
        })
    write_dataset_json(output, len(manifest))
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "crop_qc.csv").open("w", newline="") as stream:
        fields = ["case_id", "expanded_for_label", "voxels_before", "voxels_after", "ok"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"case_id": row["case_id"], "expanded_for_label": row["expanded_for_label"],
                          "voxels_before": row["detector_label_qc"]["voxels_before"],
                          "voxels_after": row["detector_label_qc"]["voxels_after"],
                          "ok": row["detector_label_qc"]["ok"]} for row in manifest)
    expanded_count = sum(row["expanded_for_label"] for row in manifest)
    print(f"Wrote {len(manifest)} cases to {output} ({expanded_count} detector crops expanded for labels).")


def main() -> None:
    try:
        build(parser().parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
