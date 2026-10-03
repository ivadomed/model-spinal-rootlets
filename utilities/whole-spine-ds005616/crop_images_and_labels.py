"""
Crop original images along the z-axis based on the extent of the segmentation (BIDS dataset):
    zmin = seg_start - below
    zmax = seg_start + above

Outputs are saved next to their inputs:
    cropped image        -> sub-XXX/anat/<image>_crop.nii.gz
    cropped segmentation -> derivatives/labels/sub-XXX/anat/<seg>_crop.nii.gz

Example:
    sct_python crop_by_seg.py -i /path/to/dataset
    sct_python crop_by_seg.py -i /path/to/dataset -seg-root /path/to/dataset/derivatives/labels \
        -above 60 -below 200 --start-at-min
"""

import argparse
import subprocess
from pathlib import Path
import numpy as np
from spinalcordtoolbox.image import Image

# ----------------------------- global settings -------------------------------
SEG_SUFFIX = '_label-SC_seg.nii.gz'   # suffix of segmentation files
IMG_SUFFIX = '.nii.gz'                # original image = seg name with SEG_SUFFIX -> IMG_SUFFIX
                                      # e.g. sub-001_T2w_label-SC_seg.nii.gz -> sub-001_T2w.nii.gz
# ------------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(
        description="Crop BIDS images along the z-axis around the start of the segmentation "
                    "using sct_crop_image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', '--bids-root', type=Path, required=True,
                        help="BIDS dataset root (contains sub-XXX/).")
    parser.add_argument('-seg-root', type=Path, default=None,
                        help="Folder with segmentations in BIDS structure. "
                             "Default: <bids-root>/derivatives/labels")
    parser.add_argument('-above', type=int, default=60,
                        help="Number of slices added beyond the segmentation start (zmax = start + above).")
    parser.add_argument('-below', type=int, default=200,
                        help="Number of slices from the segmentation start the other way (zmin = start - below).")
    parser.add_argument('--start-at-min', action='store_true',
                        help="Define segmentation start as the lowest z index (default: highest z index).")
    parser.add_argument('--no-crop-seg', action='store_true',
                        help="Do not crop the segmentation, only the original image.")
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


def crop(in_file, out_file, zmin, zmax):
    command = [
        'sct_crop_image',
        '-i', str(in_file),
        '-o', str(out_file),
        '-zmin', str(zmin),
        '-zmax', str(zmax),
    ]
    print(' '.join(command))
    subprocess.run(command, check=True)


def main():
    args = get_parser().parse_args()

    bids_root = args.bids_root
    seg_root = args.seg_root or bids_root / 'derivatives' / 'labels'

    # finds sub-XXX/anat/... and also sub-XXX/ses-XX/anat/...
    seg_files = sorted(seg_root.glob(f"sub-*/**/anat/*{SEG_SUFFIX}"))
    if args.sub:
        seg_files = [f for f in seg_files if f.relative_to(seg_root).parts[0] in args.sub]
    print(f"Found {len(seg_files)} segmentations")

    for seg_file in seg_files:
        rel_dir = seg_file.parent.relative_to(seg_root)       # e.g. sub-001/anat
        subject = rel_dir.parts[0]
        img_name = seg_file.name.replace(SEG_SUFFIX, IMG_SUFFIX)
        img_file = bids_root / rel_dir / img_name

        if not img_file.is_file():
            print(f"[{subject}] original image not found: {img_file} -> skipped")
            continue

        seg_start, z_dim = get_seg_start_z(seg_file, args.start_at_min)
        if seg_start is None:
            print(f"[{subject}] empty segmentation -> skipped")
            continue

        # bounds, clipped to the image (sct_crop_image indices are 0-based and inclusive)
        zmin_req = seg_start - args.below
        zmax_req = seg_start + args.above
        zmin = max(zmin_req, 0)
        zmax = min(zmax_req, z_dim - 1)
        if (zmin, zmax) != (zmin_req, zmax_req):
            print(f"[{subject}] warning: crop window clipped to image bounds "
                  f"(requested {zmin_req}-{zmax_req}, z_dim={z_dim})")

        print(f"[{subject}] seg start z={seg_start}, cropping z={zmin}..{zmax}")

        # cropped original image -> same folder as the original image
        crop(img_file, img_file.parent / img_name.replace('.nii.gz', '_crop.nii.gz'), zmin, zmax)

        # cropped segmentation -> same folder as the segmentation (derivatives/labels)
        if not args.no_crop_seg:
            crop(seg_file, seg_file.parent / seg_file.name.replace('.nii.gz', '_crop.nii.gz'), zmin, zmax)


if __name__ == '__main__':
    main()