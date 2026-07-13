#!/usr/bin/env python3
"""Reorient a dataset produced by ``prepare_dataset.py`` to RPI.

Reorientation only permutes and flips voxel axes; it does not resample or
interpolate image intensities or labels. Files that are already in RPI are
left untouched. By default files are changed in place. Pass ``--output`` to
create a reoriented copy instead.

Examples
--------
Reorient a dataset in place::

    python training/cervical_cropped/reorient_dataset_rpi.py \
      /scratch/$USER/nnUNet_raw/Dataset401_CervicalRootletsCropped

Create a separate reoriented dataset::

    python training/cervical_cropped/reorient_dataset_rpi.py \
      /path/to/Dataset401_CervicalRootletsCropped \
      --output /path/to/Dataset402_CervicalRootletsCroppedRPI
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


# SCT reports orientation using its historical "from" convention, which is the
# opposite of NiBabel's positive-axis convention. Thus SCT RPI (the convention
# used by the rootlets model and ``sct_image -setorient RPI``) is NiBabel LAS.
TARGET_NIB_AXCODES = ("L", "A", "S")
TARGET_ORIENTATION = axcodes2ornt(TARGET_NIB_AXCODES)
OPPOSITE_AXCODE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="Dataset directory containing imagesTr/ and labelsTr/.")
    p.add_argument("--output", type=Path,
                   help="Write a copied dataset here instead of modifying the input dataset in place.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing --output directory. Has no effect for in-place conversion.")
    p.add_argument("--dry-run", action="store_true", help="Validate and report changes without writing files.")
    return p


def nifti_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.nii.gz")) + sorted(directory.glob("*.nii"))


def case_id_from_image(path: Path) -> str:
    name = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    stem, separator, channel = name.rpartition("_")
    if not separator or len(channel) != 4 or not channel.isdigit():
        raise ValueError(f"Unexpected nnU-Net image filename: {path.name}")
    return stem


def label_for_case(labels_dir: Path, case_id: str) -> Path:
    candidates = [labels_dir / f"{case_id}.nii.gz", labels_dir / f"{case_id}.nii"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one label for {case_id}, found {len(matches)}")
    return matches[0]


def orientation(img: nib.spatialimages.SpatialImage) -> str:
    """Return orientation using SCT's convention, not NiBabel's convention."""
    return "".join(OPPOSITE_AXCODE[code] for code in nib.aff2axcodes(img.affine))


def reorient_rpi(img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    transform = ornt_transform(io_orientation(img.affine), TARGET_ORIENTATION)
    return img.as_reoriented(transform)


def same_grid(image: nib.spatialimages.SpatialImage, label: nib.spatialimages.SpatialImage) -> bool:
    return image.shape[:3] == label.shape[:3] and np.allclose(image.affine, label.affine, atol=1e-4)


def save_atomically(img: nib.spatialimages.SpatialImage, destination: Path) -> None:
    suffix = ".nii.gz" if destination.name.endswith(".nii.gz") else ".nii"
    temporary = destination.with_name(f".{destination.name}.rpi-tmp{suffix}")
    try:
        nib.save(img, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_dataset(dataset: Path) -> list[tuple[Path, Path]]:
    images_dir, labels_dir = dataset / "imagesTr", dataset / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected imagesTr/ and labelsTr/ in {dataset}")

    images = nifti_files(images_dir)
    labels = nifti_files(labels_dir)
    if not images:
        raise ValueError(f"No NIfTI images found in {images_dir}")

    pairs = [(image, label_for_case(labels_dir, case_id_from_image(image))) for image in images]
    paired_labels = {label for _, label in pairs}
    unpaired_labels = set(labels) - paired_labels
    if unpaired_labels:
        raise ValueError(f"Labels without matching images: {sorted(path.name for path in unpaired_labels)}")

    for image_path, label_path in pairs:
        image, label = nib.load(image_path), nib.load(label_path)
        if not same_grid(image, label):
            raise ValueError(f"Image and label are not on the same grid: {image_path.name}, {label_path.name}")
    return pairs


def prepare_destination(source: Path, output: Path | None, overwrite: bool, dry_run: bool) -> Path:
    if output is None:
        return source
    destination = output.resolve()
    if destination == source:
        return source
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} exists; pass --overwrite to replace it")
        if not dry_run:
            shutil.rmtree(destination)
    if not dry_run:
        shutil.copytree(source, destination)
    return destination


def run(args: argparse.Namespace) -> None:
    source = args.dataset.resolve()
    pairs = validate_dataset(source)
    destination = prepare_destination(source, args.output, args.overwrite, args.dry_run)

    changed = 0
    skipped = 0
    # A label can be shared by multiple image channels, so process every file once.
    relative_files = sorted({path.relative_to(source) for pair in pairs for path in pair})
    for index, relative_path in enumerate(relative_files, start=1):
        source_path = source / relative_path
        destination_path = destination / relative_path
        img = nib.load(source_path)
        current = orientation(img)
        if current == "RPI":
            skipped += 1
            print(f"[{index}/{len(relative_files)}] {relative_path}: already RPI; skipping")
            continue

        changed += 1
        print(f"[{index}/{len(relative_files)}] {relative_path}: {current} -> RPI")
        if not args.dry_run:
            save_atomically(reorient_rpi(img), destination_path)

    if args.dry_run:
        print(
            f"Dry run: {changed} of {len(relative_files)} files require reorientation; "
            f"{skipped} already-RPI files would be skipped."
        )
        return

    destination_pairs = validate_dataset(destination)
    for image_path, label_path in destination_pairs:
        image, label = nib.load(image_path), nib.load(label_path)
        if orientation(image) != "RPI" or orientation(label) != "RPI":
            raise RuntimeError(f"RPI verification failed for {image_path.name} and {label_path.name}")
        if not same_grid(image, label):
            raise RuntimeError(f"Grid verification failed for {image_path.name} and {label_path.name}")
    print(
        f"Reoriented {changed} of {len(relative_files)} files; skipped {skipped} already-RPI files; "
        f"verified {len(destination_pairs)} pairs in {destination}."
    )


def main() -> None:
    try:
        run(parser().parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
