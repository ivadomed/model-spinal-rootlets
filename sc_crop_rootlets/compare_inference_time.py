"""
Inference time of RootletSeg (Dataset037, run on the original image) vs RootletSeg + sc-crop (Dataset052: sc_crop.py,
nnUNetv2_predict on the cropped image, pad_predictions.py), on the same testing images, one image at a time (nnU-Net on the Apple GPU, mps).
Each step is the usual command, timed as a whole (start of the command and loading of the model included).

Output: OUTPUT_DIR/inference_time.csv: time (s) of each step for each image, then the mean and SD across the images.

Example:
    python sc_crop_rootlets/compare_inference_time.py --images <imagesTs> --models <folder with Dataset037_.../ and
        Dataset052_.../> --sc-crop-python <python of the sc_crop environment> --output-dir inference_time
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SC_CROP_REPO = HERE.parents[1] / "sc-crop"   # sc-crop repository next to this one (sc_crop.py and the YOLO weights)
SC_CROP_ARGS = ["--structure", "canal", "--conf", "0.26", "0.29", "--margin", "10", "10", "30"]   # as for Dataset052
EXT = ".nii.gz"


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path, help="Testing images (<case>_0000.nii.gz).")
    parser.add_argument("--models", required=True, type=Path, help="Folder with the models (used as nnUNet_results).")
    parser.add_argument("--datasets", nargs=2, default=["37", "52"], metavar=("ORIGINAL", "CROP"),
                        help="Model run on the original images, model run on the cropped images (default: 37 52).")
    parser.add_argument("--trainers", nargs=2, default=["nnUNetTrainer_2000epochs", "nnUNetTrainer"],
                        metavar=("ORIGINAL", "CROP"), help="Trainers of the two models.")
    parser.add_argument("--sc-crop-python", default=sys.executable, help="Python of the sc_crop environment.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output folder.")
    return parser


def run(cmd, env=None):
    """Run a command (its output in the terminal) and return its duration (s)."""
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    result = subprocess.run(cmd, env=env)
    duration = time.perf_counter() - start
    if result.returncode != 0:
        script = cmd[1] if cmd[1].endswith(".py") else cmd[0]
        raise RuntimeError(f"{Path(script).name} failed (exit code {result.returncode}, see its output above)")
    return duration


def predict(dataset, trainer, input_dir, output_dir, env):
    return run(["nnUNetv2_predict", "-d", dataset, "-tr", trainer, "-c", "3d_fullres", "-f", "all",
                "-i", input_dir, "-o", output_dir, "-device", "mps"], env)


def time_image(image, args, env):
    case, name = image.name.removesuffix(f"_0000{EXT}"), image.name.removesuffix(EXT)
    work = args.output_dir / "work" / case
    shutil.rmtree(work, ignore_errors=True)
    original, crop, cropped = work / "original", work / "crop", work / "cropped"
    for d in (original, crop, cropped):
        d.mkdir(parents=True)
    os.symlink(image.resolve(), original / image.name)
    a, b = args.labels
    t = {"case": case}

    # Model on the original image
    t[f"{a}_predict_s"] = predict(args.datasets[0], args.trainers[0], original, work / "pred_original", env)

    # Model on the cropped image: crop, predict, pad the prediction back to the original image
    t[f"{b}_crop_s"] = run([args.sc_crop_python, SC_CROP_REPO / "sc_crop.py", "--image", image,
                            "--model", SC_CROP_REPO / "model_detect_canal_sc.pt", *SC_CROP_ARGS, "--output-dir", crop])
    (crop / f"{name}_crop{EXT}").rename(cropped / image.name)                  # name expected by nnU-Net
    (crop / f"{name}_cropbox{EXT}").rename(crop / f"{case}_cropbox{EXT}")      # name expected by pad_predictions.py
    t[f"{b}_predict_s"] = predict(args.datasets[1], args.trainers[1], cropped, work / "pred_crop", env)
    t[f"{b}_pad_s"] = run([sys.executable, HERE / "pad_predictions.py", "--pred-root", work / "pred_crop",
                           "--cropbox-dir", crop, "--output-root", work / "pred_crop_padded"])
    t[f"{b}_total_s"] = t[f"{b}_crop_s"] + t[f"{b}_predict_s"] + t[f"{b}_pad_s"]
    return t


def main():
    args = get_parser().parse_args()
    args.labels = [f"Dataset{int(d):03d}" if d.isdigit() else d for d in args.datasets]
    env = {**os.environ, "nnUNet_results": str(args.models.resolve())}
    images = sorted(args.images.glob(f"*_0000{EXT}"))

    rows = []
    for i, image in enumerate(images, 1):
        print(f"\n========== [{i}/{len(images)}] {image.name} ==========", flush=True)
        try:
            rows.append(time_image(image, args, env))
            print(f"[{i}/{len(images)}] " + ", ".join(f"{k} {v:.1f}" if k != "case" else v for k, v in rows[-1].items()))
        except Exception as e:   # keep going with the other images
            rows.append({"case": image.name.removesuffix(f"_0000{EXT}"), "error": str(e)})
            print(f"[{i}/{len(images)}] {rows[-1]['case']}: ERROR {e}")

    ok = [r for r in rows if "error" not in r]
    columns = [k for k in ok[0] if k != "case"] if ok else []
    stats = [{"case": "mean", **{k: np.mean([r[k] for r in ok]) for k in columns}},
             {"case": "SD", **{k: np.std([r[k] for r in ok], ddof=1) for k in columns}}] if ok else []

    out = args.output_dir / "inference_time.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", *columns, "error"])
        writer.writeheader()
        writer.writerows({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows + stats)

    print(f"\nMean ± SD over {len(ok)}/{len(rows)} images:")
    for k in columns:
        print(f"  {k}: {stats[0][k]:.2f} ± {stats[1][k]:.2f} s")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
