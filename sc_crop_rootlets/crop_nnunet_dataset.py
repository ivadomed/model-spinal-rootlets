"""
Crop an nnU-Net dataset around the spinal canal (default) or the spinal cord by calling sc_crop.py on each image.

For each case (imagesTr/<case>_0000.nii.gz + labelsTr/<case>.nii.gz, and imagesTs/labelsTs if present):
    1. sc_crop.py is run on the channel 0 image (e.g. T2w): it saves the cropped image and the box mask
       (<case>_0000_cropbox.nii.gz)
    2. the label (and the other channels, if any) are cropped with the SAME box, read from the box mask, so images
       and labels stay aligned voxel-to-voxel

sc_crop.py is called as it is (not modified).

Output: a new nnU-Net dataset with the same structure (dataset.json and other top-level files copied):
    <OUTPUT_DIR>/imagesTr/<case>_0000.nii.gz, <OUTPUT_DIR>/labelsTr/<case>.nii.gz, ...
    <OUTPUT_DIR>/cropbox/<case>_cropbox.nii.gz : box mask on the grid of the original image (FSLeyes overlay)
    <OUTPUT_DIR>/crop_log.csv                  : box of each case, label voxels lost by the crop, errors

Example:
    python crop_nnunet_dataset.py --dataset nnUNet_raw/Dataset101_DCM --output-dir nnUNet_raw/Dataset102_DCMcrop \
        --model model_detect_canal_sc.pt --structure canal --conf 0.26 0.29 --margin 10 10 30
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

SC_CROP_REPO = Path(__file__).resolve().parents[2] / "sc-crop"   # default: sc-crop repository next to this one
SPLITS = [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]


def get_parser():
    parser = argparse.ArgumentParser(description="Crop all the images and labels of an nnU-Net dataset with sc_crop.py.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="nnU-Net dataset folder (contains imagesTr).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder of the cropped nnU-Net dataset.")
    parser.add_argument("--sc-crop-repo", type=Path, default=SC_CROP_REPO, help="sc-crop repository (sc_crop.py).")
    parser.add_argument("--model", required=True, type=Path, help="YOLO weights, passed to sc_crop.py.")
    parser.add_argument("--structure", choices=["canal", "sc"], default="canal", help="Passed to sc_crop.py.")
    parser.add_argument("--rgb", action="store_true", help="Passed to sc_crop.py.")
    parser.add_argument("--conf", nargs="+", default=["0.26", "0.29"], help="Passed to sc_crop.py (SC canal).")
    parser.add_argument("--margin", nargs="+", default=["10", "10", "30"], help="Passed to sc_crop.py (RL AP SI, mm).")
    parser.add_argument("--channel", default="0000", help="Channel suffix of the image sc_crop.py is run on.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute cases already in the output folder.")
    return parser


def find_cases(images_dir, channel):
    """Return {case: [channel images]} of an nnU-Net images folder (cases that have the detection channel)."""
    cases = {}
    for path in sorted(images_dir.glob("*.nii.gz")):
        case = path.name.removesuffix(".nii.gz").rpartition("_")[0]
        cases.setdefault(case, []).append(path)
    return {c: p for c, p in cases.items() if any(x.name == f"{c}_{channel}.nii.gz" for x in p)}


def box_from_mask(mask_path):
    """Return the slices (x, y, z) of the box saved by sc_crop.py (<name>_cropbox.nii.gz), and its image."""
    mask = nib.load(mask_path)
    idx = np.nonzero(np.asarray(mask.dataobj))
    if len(idx[0]) == 0:
        raise RuntimeError(f"empty box mask {mask_path.name}")
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx), mask


def same_grid(a, b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-3)


def run_sc_crop(image, tmp_dir, args):
    """Run sc_crop.py on `image` (outputs in tmp_dir). Return (cropped image, box mask)."""
    cmd = [sys.executable, str(args.sc_crop_repo.expanduser().resolve() / "sc_crop.py"), "--image", str(image), "--model", str(args.model),
           "--structure", args.structure, "--conf", *args.conf, "--margin", *args.margin, "--output-dir", str(tmp_dir)]
    if args.rgb:
        cmd.append("--rgb")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stderr.strip() or result.stdout.strip()).splitlines()
        raise RuntimeError(f"sc_crop.py failed: {msg[-1] if msg else result.returncode}")
    name = image.name.removesuffix(".gz").removesuffix(".nii")
    return tmp_dir / f"{name}_crop.nii.gz", tmp_dir / f"{name}_cropbox.nii.gz"


def crop_case(case, images, label_path, out_images, out_labels, out_box, args):
    ref_path = next(p for p in images if p.name == f"{case}_{args.channel}.nii.gz")
    with tempfile.TemporaryDirectory() as tmp:
        crop_path, mask_path = run_sc_crop(ref_path, Path(tmp), args)
        crop, mask = box_from_mask(mask_path)
        shutil.move(crop_path, out_images / ref_path.name)
        shutil.move(mask_path, out_box / f"{case}_cropbox.nii.gz")

    for path in images:   # other channels (if any): same box
        if path == ref_path:
            continue
        img = nib.load(path)
        if not same_grid(img, mask):
            raise RuntimeError(f"{path.name} is not on the grid of {ref_path.name}")
        nib.save(img.slicer[crop], out_images / path.name)

    lost = ""
    if label_path is not None:
        label = nib.load(label_path)
        if not same_grid(label, mask):
            raise RuntimeError(f"{label_path.name} is not on the grid of {ref_path.name}")
        data = np.asanyarray(label.dataobj)
        lost = int(np.count_nonzero(data) - np.count_nonzero(data[crop]))   # label voxels outside the box
        nib.save(label.slicer[crop], out_labels / label_path.name)

    return {"shape_in": "x".join(map(str, mask.shape[:3])),
            "shape_out": "x".join(str(s.stop - s.start) for s in crop),
            "box": " ".join(f"{s.start}-{s.stop - 1}" for s in crop), "label_voxels_lost": lost}


def main():
    args = get_parser().parse_args()
    if not (args.sc_crop_repo.expanduser() / "sc_crop.py").is_file():
        raise SystemExit(f"--sc-crop-repo: no sc_crop.py in {args.sc_crop_repo}")
    if not (args.dataset / "imagesTr").is_dir():
        raise SystemExit(f"{args.dataset} is not an nnU-Net dataset (no imagesTr folder)")
    if args.output_dir.resolve() == args.dataset.resolve():
        raise SystemExit("--output-dir must differ from --dataset (the originals would be overwritten)")
    out_box = args.output_dir / "cropbox"
    out_box.mkdir(parents=True, exist_ok=True)

    for f in args.dataset.iterdir():   # dataset.json and other top-level files
        if f.is_file():
            shutil.copy2(f, args.output_dir / f.name)

    rows = []
    for images_name, labels_name in SPLITS:
        images_dir, labels_dir = args.dataset / images_name, args.dataset / labels_name
        if not images_dir.is_dir():
            continue
        out_images, out_labels = args.output_dir / images_name, args.output_dir / labels_name
        out_images.mkdir(exist_ok=True)
        if labels_dir.is_dir():
            out_labels.mkdir(exist_ok=True)
        cases = find_cases(images_dir, args.channel)
        for i, (case, images) in enumerate(cases.items(), 1):
            tag = f"[{images_name} {i}/{len(cases)}] {case}:"
            label_path = labels_dir / f"{case}.nii.gz"
            if not label_path.exists():
                if labels_name == "labelsTr":
                    print(f"{tag} WARNING no label in {labels_name}")
                label_path = None
            row = {"split": images_name, "case": case}
            done = all((out_images / p.name).exists() for p in images) and \
                (label_path is None or (out_labels / label_path.name).exists())
            if done and not args.overwrite:
                print(f"{tag} already cropped, skipped")
                rows.append({**row, "status": "skipped"})
                continue
            try:
                info = crop_case(case, images, label_path, out_images, out_labels, out_box, args)
                rows.append({**row, "status": "ok", **info})
                warn = f"  WARNING {info['label_voxels_lost']} label voxels outside the box" \
                    if info["label_voxels_lost"] not in ("", 0) else ""
                print(f"{tag} {info['shape_in']} -> {info['shape_out']}{warn}")
            except Exception as e:   # keep going with the other cases
                rows.append({**row, "status": f"error: {e}"})
                print(f"{tag} ERROR {e}")

    fields = ["split", "case", "status", "shape_in", "shape_out", "box", "label_voxels_lost"]
    log_path = args.output_dir / "crop_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    errors = [r for r in rows if r["status"].startswith("error")]
    lost = [r for r in rows if r.get("label_voxels_lost") not in (None, "", 0)]
    print(f"\nDone: {len(rows)} cases, {len(errors)} errors, {len(lost)} with label voxels outside the box. "
          f"Log: {log_path}")
    if errors:
        print("Cases with errors are NOT in the output dataset: fix them, or update numTraining in dataset.json.")


if __name__ == "__main__":
    main()
