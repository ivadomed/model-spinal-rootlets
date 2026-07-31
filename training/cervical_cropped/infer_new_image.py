#!/usr/bin/env python3
"""Run the Dataset403 cropped fold_all model on one full NIfTI image."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


PADDING_MM = {
    "pad_superior": 50.0,
    "pad_inferior": 110.0,
    "pad_left": 25.0,
    "pad_right": 25.0,
    "pad_anterior": 40.0,
    "pad_posterior": 32.0,
}
MODEL_ORIENTATION = axcodes2ornt(("L", "A", "S"))  # SCT's RPI convention.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model",
        help="Folder containing dataset.json, plans.json, and fold_all/.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device for nnU-Net inference.",
    )
    parser.add_argument(
        "--crop-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for sc-crop detection.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reorient(
    image: nib.spatialimages.SpatialImage,
    target: np.ndarray,
) -> nib.spatialimages.SpatialImage:
    transform = ornt_transform(io_orientation(image.affine), target)
    return image.as_reoriented(transform)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
    for relative in ("dataset.json", "plans.json", "fold_all/checkpoint_best.pth"):
        path = args.model_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
    if "cuda" in {args.device, args.crop_device} and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install CUDA PyTorch or pass --device cpu")


def predict(args: argparse.Namespace) -> None:
    from sc_crop import crop, detect, uncrop

    validate_args(args)
    input_image = nib.load(args.input)
    bbox = detect(input_image, device=args.crop_device, **PADDING_MM)
    cropped_native = crop(input_image, bbox)
    native_crop_orientation = io_orientation(cropped_native.affine)
    cropped_rpi = reorient(cropped_native, MODEL_ORIENTATION)

    with tempfile.TemporaryDirectory(prefix="rootlets-cropped-") as temporary:
        temporary = Path(temporary)
        images = temporary / "images"
        predictions = temporary / "predictions"
        images.mkdir()
        predictions.mkdir()
        nib.save(cropped_rpi, images / "case_0000.nii.gz")

        for variable in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            directory = temporary / variable
            directory.mkdir()
            os.environ.setdefault(variable, str(directory))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = torch.device(args.device)
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=args.device == "cuda",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(args.model_dir),
            use_folds=("all",),
            checkpoint_name="checkpoint_best.pth",
        )
        predictor.predict_from_files(
            str(images),
            str(predictions),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        prediction_rpi = nib.load(predictions / "case.nii.gz")
        prediction_native = reorient(prediction_rpi, native_crop_orientation)
        if prediction_native.shape[:3] != cropped_native.shape[:3] or not np.allclose(
            prediction_native.affine, cropped_native.affine, atol=2e-4
        ):
            raise RuntimeError("Prediction does not match the native crop grid")

        prediction_full = uncrop(prediction_native, bbox)
        prediction_full.header.set_data_dtype(np.uint8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(prediction_full, args.output)

    labels = sorted(
        int(value)
        for value in np.unique(np.asanyarray(prediction_full.dataobj))
        if value != 0
    )
    print(f"Output: {args.output}")
    print(f"Labels: {labels}")


def main() -> None:
    predict(parse_args())


if __name__ == "__main__":
    main()
