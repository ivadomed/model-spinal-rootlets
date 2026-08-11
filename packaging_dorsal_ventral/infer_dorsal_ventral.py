#!/usr/bin/env python3
"""Classify an existing level-labelled rootlet mask as dorsal or ventral."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


RPI = axcodes2ornt(("R", "P", "I"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--image", required=True, type=Path)
    parser.add_argument("-r", "--rootlets", required=True, type=Path)
    parser.add_argument("--output-class-map", required=True, type=Path)
    parser.add_argument("--output-dorsal", required=True, type=Path)
    parser.add_argument("--output-ventral", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
        help="Folder containing dataset.json, plans.json, and fold_0/.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    for path in (args.image, args.rootlets):
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
    outputs = (args.output_class_map, args.output_dorsal, args.output_ventral)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {existing[0]}; pass --overwrite to replace outputs"
        )
    for relative in ("dataset.json", "plans.json", "fold_0/checkpoint_final.pth"):
        path = args.model_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install CUDA PyTorch or use --device cpu")

    image = nib.load(args.image)
    rootlets = nib.load(args.rootlets)
    if image.shape != rootlets.shape or not np.allclose(
        image.affine, rootlets.affine, atol=1e-4
    ):
        raise ValueError("MRI and rootlet labels must use the same voxel grid")
    values = np.asanyarray(rootlets.dataobj)
    if not np.all(np.isfinite(values)) or not np.array_equal(values, np.rint(values)):
        raise ValueError("Rootlet labels must be finite integers")
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("Rootlet labels must be non-negative and non-empty")
    return image, rootlets


def _reorient(
    image: nib.spatialimages.SpatialImage, target: np.ndarray
) -> nib.spatialimages.SpatialImage:
    return image.as_reoriented(ornt_transform(io_orientation(image.affine), target))


def _probabilities(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    archive = np.load(path)
    data = np.asarray(archive["probabilities"])
    if data.shape[1:] == shape:
        return data
    if data.shape[1:] == tuple(reversed(shape)):
        return data.transpose(0, 3, 2, 1)
    raise ValueError("Probability array does not match the nnU-Net output grid")


def _partition(
    rootlets: np.ndarray, segmentation: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.rint(rootlets).astype(np.int16)
    support = labels > 0
    classes = np.rint(segmentation).astype(np.uint8)
    invalid = support & ~np.isin(classes, (1, 2))
    if np.any(invalid):
        classes[invalid] = (
            1 + np.argmax(probabilities[1:3], axis=0).astype(np.uint8)
        )[invalid]
    class_map = np.zeros(labels.shape, dtype=np.uint8)
    class_map[support] = classes[support]
    dorsal = np.where(class_map == 1, labels, 0).astype(np.int16)
    ventral = np.where(class_map == 2, labels, 0).astype(np.int16)
    if not np.array_equal(dorsal + ventral, labels):
        raise RuntimeError("Classifier did not preserve the supplied rootlet support")
    return class_map, dorsal, ventral


def _save_native(
    data_rpi: np.ndarray,
    original: nib.Nifti1Image,
    path: Path,
    dtype: np.dtype,
) -> None:
    transform = ornt_transform(RPI, io_orientation(original.affine))
    native = nib.orientations.apply_orientation(data_rpi, transform)
    if native.shape != original.shape:
        raise RuntimeError("Output shape differs from the original rootlet grid")
    header = original.header.copy()
    header.set_data_dtype(dtype)
    output = nib.Nifti1Image(native.astype(dtype), original.affine, header=header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, path)


def predict(args: argparse.Namespace) -> None:
    image, rootlet_image = _validate(args)
    image_rpi = _reorient(image, RPI)
    rootlets_rpi = _reorient(rootlet_image, RPI)
    if image_rpi.shape != rootlets_rpi.shape:
        raise RuntimeError("RPI conversion changed the paired input shapes")

    with tempfile.TemporaryDirectory(prefix="rootlet-dv-") as temporary_name:
        temporary = Path(temporary_name)
        inputs = temporary / "inputs"
        predictions = temporary / "predictions"
        inputs.mkdir()
        predictions.mkdir()
        reference_affine = image_rpi.affine
        nib.save(
            nib.Nifti1Image(
                np.asanyarray(image_rpi.dataobj).astype(np.float32), reference_affine
            ),
            inputs / "case_0000.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(
                np.rint(np.asanyarray(rootlets_rpi.dataobj)).astype(np.int16),
                reference_affine,
            ),
            inputs / "case_0001.nii.gz",
        )

        for variable in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            directory = temporary / variable
            directory.mkdir()
            os.environ.setdefault(variable, str(directory))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=args.device == "cuda",
            device=torch.device(args.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(args.model_dir),
            use_folds=(0,),
            checkpoint_name="checkpoint_final.pth",
        )
        predictor.predict_from_files(
            str(inputs),
            str(predictions),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        segmentation_image = nib.load(predictions / "case.nii.gz")
        if segmentation_image.shape != rootlets_rpi.shape:
            raise RuntimeError("nnU-Net output differs from the RPI rootlet grid")
        segmentation = np.asanyarray(segmentation_image.dataobj)
        probabilities = _probabilities(predictions / "case.npz", rootlets_rpi.shape)
        class_map, dorsal, ventral = _partition(
            np.asanyarray(rootlets_rpi.dataobj), segmentation, probabilities
        )

    _save_native(class_map, rootlet_image, args.output_class_map, np.uint8)
    _save_native(dorsal, rootlet_image, args.output_dorsal, np.int16)
    _save_native(ventral, rootlet_image, args.output_ventral, np.int16)
    print(f"Class map: {args.output_class_map}")
    print(f"Dorsal: {args.output_dorsal}")
    print(f"Ventral: {args.output_ventral}")


def main() -> None:
    predict(parse_args())


if __name__ == "__main__":
    main()
