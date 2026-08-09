#!/usr/bin/env python3
"""Render qualitative overlays comparing deterministic D/V split methods.

The input manifest has one row per method and scan, with columns:
``subject,session,image,combined,method,dorsal,ventral``.  Methods for the
same subject/session must refer to the same anatomical image and RootletSeg
mask.  The renderer rejects masks that do not exactly partition that fixed
support, so the figure cannot hide a support-changing comparison.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


@dataclass(frozen=True)
class MethodMasks:
    """Validated D/V masks for one deterministic method."""

    name: str
    dorsal: np.ndarray
    ventral: np.ndarray


@dataclass(frozen=True)
class ScanComparison:
    """One scan and all method outputs compared on its fixed support."""

    subject: str
    session: str
    image: np.ndarray
    combined: np.ndarray
    methods: tuple[MethodMasks, ...]


def _resolve(value: str, directory: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else directory / path


def _canonical_data(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    return image, np.asanyarray(image.dataobj)


def _label_data(path: Path, *, description: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    image, data = _canonical_data(path)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{description} contains non-finite values: {path}")
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise ValueError(f"{description} must contain integer label values: {path}")
    return image, rounded.astype(np.int16)


def _check_grid(
    candidate: nib.Nifti1Image,
    reference: nib.Nifti1Image,
    *,
    description: str,
) -> None:
    if candidate.shape != reference.shape or not np.allclose(
        candidate.affine, reference.affine, atol=1e-4
    ):
        raise ValueError(f"{description} grid differs from the anatomical image.")


def read_comparisons(manifest: Path) -> list[ScanComparison]:
    """Load and validate a method-comparison manifest."""

    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject", "session", "image", "combined", "method", "dorsal", "ventral"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Manifest is missing comparison columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Method-comparison manifest has no rows.")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not all(row[field].strip() for field in required):
            raise ValueError("Method-comparison manifest contains an empty required value.")
        grouped[(row["subject"], row["session"])].append(row)

    comparisons: list[ScanComparison] = []
    for (subject, session), scan_rows in sorted(grouped.items()):
        image_paths = {
            _resolve(row["image"], manifest.parent).resolve() for row in scan_rows
        }
        combined_paths = {
            _resolve(row["combined"], manifest.parent).resolve() for row in scan_rows
        }
        if len(image_paths) != 1 or len(combined_paths) != 1:
            raise ValueError(
                f"{subject} {session} must use one image and one fixed combined mask."
            )
        image_path = image_paths.pop()
        combined_path = combined_paths.pop()
        image_nii, anatomical = _canonical_data(image_path)
        combined_nii, combined = _label_data(combined_path, description="Combined mask")
        _check_grid(combined_nii, image_nii, description="Combined mask")

        methods: list[MethodMasks] = []
        names: set[str] = set()
        for row in scan_rows:
            name = row["method"]
            if name in names:
                raise ValueError(f"Duplicate method {name!r} for {subject} {session}.")
            names.add(name)
            dorsal_nii, dorsal = _label_data(
                _resolve(row["dorsal"], manifest.parent), description=f"{name} dorsal mask"
            )
            ventral_nii, ventral = _label_data(
                _resolve(row["ventral"], manifest.parent), description=f"{name} ventral mask"
            )
            _check_grid(dorsal_nii, image_nii, description=f"{name} dorsal mask")
            _check_grid(ventral_nii, image_nii, description=f"{name} ventral mask")
            if np.any((dorsal > 0) & (ventral > 0)) or not np.array_equal(
                dorsal + ventral, combined
            ):
                raise ValueError(
                    f"{name} does not exactly partition the fixed RootletSeg support for "
                    f"{subject} {session}."
                )
            methods.append(MethodMasks(name=name, dorsal=dorsal, ventral=ventral))
        comparisons.append(
            ScanComparison(
                subject=subject,
                session=session,
                image=anatomical,
                combined=combined,
                methods=tuple(methods),
            )
        )
    return comparisons


def _slice_indices(combined: np.ndarray, panels: int) -> list[int]:
    support = np.flatnonzero(np.any(combined > 0, axis=(0, 1)))
    if not support.size:
        raise ValueError("Cannot render an empty combined RootletSeg mask.")
    positions = np.linspace(0, support.size - 1, panels)
    return [int(support[int(round(position))]) for position in positions]


def _comparison_slice_indices(scan: ScanComparison, panels: int) -> list[tuple[int, int]]:
    """Choose representative slices, preferring areas where methods differ."""

    support = np.flatnonzero(np.any(scan.combined > 0, axis=(0, 1)))
    if not support.size:
        raise ValueError("Cannot render an empty combined RootletSeg mask.")
    reference = scan.methods[0].dorsal > 0
    labels = np.stack([method.dorsal > 0 for method in scan.methods])
    disagreement = np.count_nonzero(np.any(labels != reference, axis=0), axis=(0, 1))
    selections: list[tuple[int, int]] = []
    for group in np.array_split(support, panels):
        scores = disagreement[group]
        best = np.flatnonzero(scores == scores.max())
        index = int(group[best[len(best) // 2]])
        selections.append((index, int(disagreement[index])))
    return selections


def _crop_limits(combined: np.ndarray, padding: int = 8) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return a fixed axial crop around RootletSeg support for readable overlays."""

    support = np.any(combined > 0, axis=2)
    x_values, y_values = np.nonzero(support)
    if not x_values.size:
        raise ValueError("Cannot crop an empty combined RootletSeg mask.")
    x_bounds = (max(0, int(x_values.min()) - padding), min(combined.shape[0], int(x_values.max()) + padding + 1))
    y_bounds = (max(0, int(y_values.min()) - padding), min(combined.shape[1], int(y_values.max()) + padding + 1))
    return x_bounds, y_bounds


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, (1, 99))
    return (float(lower), float(upper if upper > lower else lower + 1.0))


def render_comparison(scan: ScanComparison, output: Path, panels: int = 6) -> None:
    """Render a rows-by-method, shared-slice qualitative comparison panel."""

    if panels < 2:
        raise ValueError("panels must be at least 2.")
    figure, axes = plt.subplots(
        len(scan.methods), panels, figsize=(2.7 * panels, 2.8 * len(scan.methods)), squeeze=False
    )
    lower, upper = _display_limits(scan.image)
    dorsal_map = matplotlib.colors.ListedColormap([(1.0, 0.1, 0.1, 0.65)])
    ventral_map = matplotlib.colors.ListedColormap([(0.0, 0.85, 1.0, 0.65)])
    x_bounds, y_bounds = _crop_limits(scan.combined)
    slices = _comparison_slice_indices(scan, panels)
    for row_index, method in enumerate(scan.methods):
        for column, (z_index, disagreement) in enumerate(slices):
            axis = axes[row_index, column]
            background = scan.image[:, :, z_index].T
            dorsal = method.dorsal[:, :, z_index].T > 0
            ventral = method.ventral[:, :, z_index].T > 0
            axis.imshow(background, cmap="gray", origin="lower", vmin=lower, vmax=upper)
            axis.imshow(
                np.ma.masked_where(~dorsal, dorsal),
                cmap=dorsal_map,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axis.imshow(
                np.ma.masked_where(~ventral, ventral),
                cmap=ventral_map,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            if row_index == 0:
                axis.set_title(f"z={z_index} · {disagreement} differing voxels", fontsize=8)
            axis.set_xlim(x_bounds)
            axis.set_ylim(y_bounds)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.10,
                    0.5,
                    method.name,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=9,
                    fontweight="bold",
                    clip_on=False,
                )
    figure.suptitle(
        f"{scan.subject} {scan.session}: deterministic D/V method comparison\n"
        "Same RootletSeg support; qualitative QC only, not anatomical accuracy",
        fontsize=12,
    )
    figure.legend(
        handles=[
            Patch(facecolor=(1.0, 0.1, 0.1, 0.65), label="predicted dorsal"),
            Patch(facecolor=(0.0, 0.85, 1.0, 0.65), label="predicted ventral"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.04, 0.06, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(manifest: Path, output_directory: Path, panels: int = 6) -> list[Path]:
    """Render one method-comparison PNG for every scan in *manifest*."""

    outputs: list[Path] = []
    for scan in read_comparisons(manifest):
        output = output_directory / f"{scan.subject}_{scan.session}_method-comparison.png"
        render_comparison(scan, output, panels=panels)
        outputs.append(output)
    return outputs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panels", type=int, default=6)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    for output in run(Path(args.manifest), Path(args.output_dir), panels=args.panels):
        print(output)


if __name__ == "__main__":
    main()
