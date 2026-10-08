"""
Reverse of crop_by_seg.py: bring labels drawn/corrected on the cropped images back to the original
(uncropped) space.

The crop window is recomputed from the ORIGINAL (uncropped) spinal cord segmentation exactly as in crop_by_seg.py:
    zmin = seg_start - below
    zmax = seg_start + above
(clipped to the image bounds). The cropped label is zero-padded along z into the original image grid and saved
with the original image's header (affine, orientation, voxel size). Its JSON sidecar (if any) is copied unchanged
next to the output, renamed without '_crop'.

Naming (any position of '_crop' works):
    cropped label   <label-root>/.../sub-XXX_T2w_crop_label-rootlets_dseg.nii.gz   (+ .json)
                    <label-root>/.../sub-XXX_T2w_label-SC_seg_crop.nii.gz          (+ .json)
    original image  <bids-root>/sub-XXX/anat/sub-XXX_T2w.nii.gz
    SC segmentation <seg-root>/sub-XXX/anat/sub-XXX_T2w_label-SC_seg.nii.gz
    output          <out-root>/sub-XXX/anat/<label name without '_crop'>.nii.gz   (+ .json)

<label-root> may be BIDS-structured or flat; the output always mirrors the BIDS structure of <seg-root>.

IMPORTANT: use the same -above / -below / --start-at-min values as in the forward crop.

Example:
    sct_python uncrop_by_seg.py \
        -i /Users/katerinakrejci/Documents/Data/ds005616 \
        -label-root /Users/katerinakrejci/Documents/Results/whole-spine-ds005616/manual-corrections \
        -o /Users/katerinakrejci/Documents/Results/whole-spine-ds005616/manual-corrections-uncropped \
        -above 60 -below 200
"""

import argparse
import shutil
from pathlib import Path
import numpy as np
from spinalcordtoolbox.image import Image

# ----------------------------- global settings -------------------------------
SEG_SUFFIX = '_label-SC_seg.nii.gz'   # suffix of the (uncropped) SC segmentation
IMG_SUFFIX = '.nii.gz'                # original image = <base> + IMG_SUFFIX
CROP_TAG = '_crop'                    # tag added by crop_by_seg.py
# ------------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pad cropped labels back to the original image space, using the crop window "
                    "recomputed from the spinal cord segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', '--bids-root', type=Path, required=True,
                        help="BIDS dataset root with the original images (contains sub-XXX/).")
    parser.add_argument('-seg-root', type=Path, default=None,
                        help="Folder with the original (uncropped) SC segmentations in BIDS structure. "
                             "Default: <bids-root>/derivatives/labels")
    parser.add_argument('-label-root', type=Path, required=True,
                        help="Folder with the cropped labels/corrections (BIDS-structured or flat).")
    parser.add_argument('-o', '--out-root', type=Path, required=True,
                        help="Output folder; uncropped labels are saved in BIDS structure (sub-XXX/anat/).")
    parser.add_argument('-label-pattern', default=f'*{CROP_TAG}*.nii.gz',
                        help="Glob pattern of the cropped label files (searched recursively).")
    parser.add_argument('-above', type=int, default=60,
                        help="Must match the forward crop (zmax = start + above).")
    parser.add_argument('-below', type=int, default=200,
                        help="Must match the forward crop (zmin = start - below).")
    parser.add_argument('--start-at-min', action='store_true',
                        help="Must match the forward crop (segmentation start = lowest z index).")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing output files.")
    parser.add_argument('-sub', nargs='+', default=None,
                        help="Process only these subjects (e.g. sub-001 sub-002).")
    return parser


def get_seg_start_z(seg_file, start_at_min=False):
    """Return the z index (native orientation) where the segmentation starts, and the z dimension."""
    seg = Image(str(seg_file))
    data = np.asarray(seg.data)
    z_idx = np.where(np.any(data > 0, axis=(0, 1)))[0]
    if z_idx.size == 0:
        return None, seg.dim[2]
    start = z_idx.min() if start_at_min else z_idx.max()
    return int(start), seg.dim[2]


def voxel_offset_from_affines(orig_img, crop_img):
    """Position of the cropped image's first voxel in the original voxel grid (sanity check)."""
    orig_aff = orig_img.hdr.get_best_affine()
    crop_aff = crop_img.hdr.get_best_affine()
    return (np.linalg.inv(orig_aff) @ crop_aff @ np.array([0, 0, 0, 1]))[:3]


def base_name(label_name):
    """'sub-001_T2w_crop_label-rootlets_dseg.nii.gz' or 'sub-001_T2w_label-SC_seg_crop.nii.gz' -> 'sub-001_T2w'."""
    stem = label_name.replace(CROP_TAG, '', 1)
    stem = stem[:-len('.nii.gz')] if stem.endswith('.nii.gz') else stem
    return stem.split('_label-')[0]


def main():
    args = get_parser().parse_args()

    bids_root = args.bids_root
    seg_root = args.seg_root or bids_root / 'derivatives' / 'labels'
    label_root = args.label_root
    out_root = args.out_root

    label_files = sorted(label_root.rglob(args.label_pattern))
    if args.sub:
        label_files = [f for f in label_files if f.name.split('_')[0] in args.sub]
    print(f"Found {len(label_files)} cropped labels in {label_root}")

    n_done = 0
    for label_file in label_files:
        subject = label_file.name.split('_')[0]
        base = base_name(label_file.name)                       # e.g. sub-001_T2w

        # locate the original SC segmentation -> gives the BIDS sub-folder (sub-XXX[/ses-XX]/anat)
        seg_matches = sorted(seg_root.glob(f"{subject}/**/anat/{base}{SEG_SUFFIX}"))
        if len(seg_matches) != 1:
            print(f"[{subject}] expected 1 SC segmentation '{base}{SEG_SUFFIX}' in {seg_root}, "
                  f"found {len(seg_matches)} -> skipped")
            continue
        seg_file = seg_matches[0]
        rel_dir = seg_file.parent.relative_to(seg_root)        # e.g. sub-001/anat

        img_file = bids_root / rel_dir / f"{base}{IMG_SUFFIX}"
        if not img_file.is_file():
            print(f"[{subject}] original image not found: {img_file} -> skipped")
            continue

        out_dir = out_root / rel_dir
        out_file = out_dir / label_file.name.replace(CROP_TAG, '', 1)
        if out_file.resolve() == seg_file.resolve():
            print(f"[{subject}] output would overwrite the original SC segmentation -> skipped")
            continue
        if out_file.exists() and not args.overwrite:
            print(f"[{subject}] output exists: {out_file} -> skipped (use --overwrite)")
            continue

        # recompute the crop window exactly as in crop_by_seg.py
        seg_start, z_dim = get_seg_start_z(seg_file, args.start_at_min)
        if seg_start is None:
            print(f"[{subject}] empty SC segmentation -> skipped")
            continue
        zmin = max(seg_start - args.below, 0)
        zmax = min(seg_start + args.above, z_dim - 1)

        orig = Image(str(img_file))
        label = Image(str(label_file))
        lab_data = np.asarray(label.data)

        # consistency checks
        expected_shape = (orig.dim[0], orig.dim[1], zmax - zmin + 1)
        if lab_data.shape[:3] != expected_shape:
            print(f"[{subject}] {label_file.name}: shape {lab_data.shape[:3]} != expected {expected_shape} "
                  f"(check -above/-below/--start-at-min) -> skipped")
            continue
        offset = voxel_offset_from_affines(orig, label)
        if not np.allclose(offset, [0, 0, zmin], atol=0.01):
            print(f"[{subject}] {label_file.name}: header offset {np.round(offset, 2)} != recomputed "
                  f"(0, 0, {zmin}) (was the SC segmentation changed after cropping?) -> skipped")
            continue

        # pad into the original grid
        out_data = np.zeros(tuple(orig.dim[:3]) + lab_data.shape[3:], dtype=lab_data.dtype)
        out_data[:, :, zmin:zmax + 1, ...] = lab_data

        out_dir.mkdir(parents=True, exist_ok=True)
        out = orig.copy()
        out.data = out_data
        out.save(str(out_file), dtype=label.hdr.get_data_dtype().name)

        # JSON sidecar: same content, name without '_crop'
        json_in = label_file.with_name(label_file.name.replace('.nii.gz', '.json'))
        if json_in.is_file():
            json_out = out_file.with_name(out_file.name.replace('.nii.gz', '.json'))
            shutil.copy2(json_in, json_out)
        else:
            print(f"[{subject}] warning: no JSON sidecar found for {label_file.name}")

        n_done += 1
        print(f"[{subject}] z={zmin}..{zmax} -> {out_file}")

    print(f"Done: {n_done}/{len(label_files)} labels uncropped into {out_root}")


if __name__ == '__main__':
    main()