"""
Pad nnU-Net predictions made on the cropped images (crop_nnunet_dataset.py) back to the grid of the original images,
so they can be compared with the original (uncropped) ground truth and with a model run on the original images.

For each prediction <case>.nii.gz, the box mask <case>_cropbox.nii.gz saved by crop_nnunet_dataset.py (on the grid of
the original image) gives the original shape, affine and the box. The prediction is written into the box of an empty
volume (0 = background outside the box).

Checks (per case):
    - the prediction has the shape of the box (else: error, case skipped)
    - the prediction affine equals the affine of the box in the original image (warning otherwise, e.g. if the
      prediction was resampled or reoriented)

Cases whose crop failed have no prediction (and no box). With --reference-dir (e.g. the ORIGINAL labelsTs), every
reference case without a padded prediction is listed, and with --empty-for-missing an empty prediction is written for
it, so that the evaluation counts it as a failure instead of silently leaving it out.

Folders: if --pred-root is one prediction folder, the padded predictions are written directly into --output-root
(no subfolder is created). If --pred-root contains several predict_fold_* folders, each is padded into
--output-root/<same name> (one folder per fold, so the cases of different folds do not overwrite each other).

Output:
    <OUTPUT_ROOT>/<case>.nii.gz        : prediction on the grid of the original image   (one prediction folder)
    <OUTPUT_ROOT>/pad_log.csv          : status of each case
    <OUTPUT_ROOT>/<fold>/...           : same, per fold                                 (predict_fold_* folders)

Example:
    python pad_predictions.py --pred-root $nnUNet_results/Dataset102_DCMcrop/.../predictions \\
        --cropbox-dir nnUNet_raw/Dataset102_DCMcrop/cropbox --output-root predictions_padded \\
        --reference-dir nnUNet_raw/Dataset101_DCM/labelsTs --empty-for-missing
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

EXT = ".nii.gz"


def get_parser():
    parser = argparse.ArgumentParser(description="Pad predictions on cropped images back to the original grid.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pred-root", required=True, type=Path,
                        help="Folder with the predictions (<case>.nii.gz), or a folder with predict_fold_* folders.")
    parser.add_argument("--cropbox-dir", required=True, type=Path,
                        help="cropbox folder of the cropped dataset (<case>_cropbox.nii.gz, crop_nnunet_dataset.py).")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Output folder of the padded predictions (written directly into it for one prediction "
                             "folder; one subfolder per predict_fold_* if --pred-root contains several).")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="ORIGINAL (uncropped) labels, e.g. labelsTs of the original dataset: used to list the "
                             "cases without a prediction, and to check the padded predictions are on their grid.")
    parser.add_argument("--empty-for-missing", action="store_true",
                        help="With --reference-dir: write an empty prediction for reference cases without one "
                             "(failed crop or prediction), so they count as failures in the evaluation.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute cases already padded.")
    return parser


def case_name(path):
    return path.name.removesuffix(EXT)


def same_grid(a, b, atol=1e-3):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol)


def box_from_mask(mask):
    """Slices (x, y, z) of the box saved by sc_crop.py (non-zero voxels of the box mask)."""
    idx = np.nonzero(np.asarray(mask.dataobj))
    if len(idx[0]) == 0:
        raise RuntimeError("empty box mask")
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def pad_case(pred_path, mask_path, out_path, reference_path=None):
    """Write the prediction into the box of an empty volume on the grid of the original image. Return a log row."""
    mask = nib.load(mask_path)
    box = box_from_mask(mask)
    pred = nib.load(pred_path)
    data = np.asanyarray(pred.dataobj)
    box_shape = tuple(s.stop - s.start for s in box)
    if data.shape[:3] != box_shape:
        raise RuntimeError(f"prediction shape {data.shape[:3]} != box shape {box_shape}")

    warnings = []
    expected_affine = mask.slicer[box].affine   # affine of the cropped image (= what nnU-Net predicted on)
    if not np.allclose(pred.affine, expected_affine, atol=1e-3):
        warnings.append("prediction affine differs from the box affine")

    padded = np.zeros(mask.shape[:3] + data.shape[3:], dtype=data.dtype)
    padded[box] = data
    header = pred.header.copy()
    header.set_data_shape(padded.shape)
    out = nib.Nifti1Image(padded, mask.affine, header)
    out.set_qform(mask.affine, code=int(mask.header["qform_code"]) or 1)
    out.set_sform(mask.affine, code=int(mask.header["sform_code"]) or 1)

    if reference_path is not None and reference_path.exists():
        if not same_grid(out, nib.load(reference_path)):
            warnings.append(f"not on the grid of {reference_path.name}")
    nib.save(out, out_path)
    return {"status": "ok", "shape_crop": "x".join(map(str, box_shape)),
            "shape_padded": "x".join(map(str, mask.shape[:3])),
            "box": " ".join(f"{s.start}-{s.stop - 1}" for s in box), "warning": "; ".join(warnings)}


def pad_folder(pred_dir, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = sorted(pred_dir.glob(f"*{EXT}"))
    print(f"\n=== {pred_dir} -> {out_dir} ({len(preds)} predictions) ===")
    rows = []
    for pred_path in preds:
        case = case_name(pred_path)
        out_path = out_dir / pred_path.name
        row = {"case": case}
        mask_path = args.cropbox_dir / f"{case}_cropbox{EXT}"
        if not mask_path.exists():
            rows.append({**row, "status": "error: no cropbox"})
            print(f"  {case}: ERROR no {mask_path.name} in {args.cropbox_dir}")
            continue
        if out_path.exists() and not args.overwrite:
            rows.append({**row, "status": "skipped (exists)"})
            continue
        reference_path = args.reference_dir / pred_path.name if args.reference_dir else None
        try:
            info = pad_case(pred_path, mask_path, out_path, reference_path)
            rows.append({**row, **info})
            print(f"  {case}: {info['shape_crop']} -> {info['shape_padded']}"
                  f"{'  WARNING ' + info['warning'] if info['warning'] else ''}")
        except Exception as e:   # keep going with the other cases
            rows.append({**row, "status": f"error: {e}"})
            print(f"  {case}: ERROR {e}")

    # Reference cases without a padded prediction (failed crop or prediction)
    if args.reference_dir:
        done = {r["case"] for r in rows if r["status"] == "ok" or r["status"].startswith("skipped")}
        missing = [p for p in sorted(args.reference_dir.glob(f"*{EXT}")) if case_name(p) not in done]
        for ref in missing:
            row = {"case": case_name(ref)}
            if args.empty_for_missing:
                img = nib.load(ref)
                empty = np.zeros(img.shape, dtype=np.uint8)
                nib.save(nib.Nifti1Image(empty, img.affine, img.header), out_dir / ref.name)
                rows.append({**row, "status": "missing: empty prediction written"})
            else:
                rows.append({**row, "status": "missing: no prediction"})
        if missing:
            print(f"  WARNING {len(missing)} reference case(s) without a prediction"
                  f"{' (empty predictions written)' if args.empty_for_missing else ' (left out of the evaluation; use --empty-for-missing to count them as failures)'}: "
                  f"{[case_name(p) for p in missing]}")

    fields = ["case", "status", "shape_crop", "shape_padded", "box", "warning"]
    with open(out_dir / "pad_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    n_err = sum(r["status"].startswith("error") for r in rows)
    n_warn = sum(bool(r.get("warning")) for r in rows)
    print(f"  Done: {n_ok} padded, {n_err} errors, {n_warn} with warnings. Log: {out_dir / 'pad_log.csv'}")


def main():
    args = get_parser().parse_args()
    if not args.cropbox_dir.is_dir():
        raise SystemExit(f"{args.cropbox_dir} not found")
    fold_dirs = sorted(d for d in args.pred_root.glob("predict_fold_*") if d.is_dir())
    if fold_dirs:
        pairs = [(d, args.output_root / d.name) for d in fold_dirs]
    else:
        pairs = [(args.pred_root, args.output_root)]
    for pred_dir, out_dir in pairs:
        if out_dir.resolve() == pred_dir.resolve():
            raise SystemExit("--output-root must differ from --pred-root (the predictions would be overwritten)")
        pad_folder(pred_dir, out_dir, args)


if __name__ == "__main__":
    main()
