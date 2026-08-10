#!/usr/bin/env python3
"""Render an anonymized qualitative review pack for fixed-support D/V outputs.

The manifest must contain ``image``, ``combined``, ``cord``, ``dorsal``, and
``ventral`` columns.  The renderer validates every D/V pair before it writes:

* a representative axial montage;
* an animated axial slice sweep; and
* a cohort QC summary that shows class balance and support preservation.

The representative scan is selected as the median dorsal-support fraction, not
by visual appearance.  These figures are review aids, not anatomical accuracy
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402


REQUIRED_COLUMNS = {"image", "combined", "cord", "dorsal", "ventral"}
DORSAL_COLOUR = (0.90, 0.15, 0.15, 0.70)
VENTRAL_COLOUR = (0.00, 0.75, 0.90, 0.70)
ROOTLET_COLOUR = (1.00, 0.72, 0.08, 0.70)
CORD_COLOUR = (1.00, 1.00, 1.00, 0.95)


@dataclass(frozen=True)
class Scan:
    """One validated fixed-support D/V prediction, with no displayed identifier."""

    image: np.ndarray
    combined: np.ndarray
    cord: np.ndarray
    dorsal: np.ndarray
    ventral: np.ndarray

    @property
    def support_voxels(self) -> int:
        return int(np.count_nonzero(self.combined))

    @property
    def dorsal_fraction(self) -> float:
        return float(np.count_nonzero(self.dorsal) / self.support_voxels)


def _resolve(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _load(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    return image, np.asanyarray(image.dataobj)


def _labels(path: Path, *, description: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    image, values = _load(path)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{description} contains non-finite values: {path}")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{description} must contain integer labels: {path}")
    return image, rounded.astype(np.int16)


def _check_grid(candidate: nib.Nifti1Image, reference: nib.Nifti1Image, *, field: str) -> None:
    if candidate.shape != reference.shape or not np.allclose(
        candidate.affine, reference.affine, atol=1e-4
    ):
        raise ValueError(f"{field} does not share the anatomical image grid.")


def read_manifest(manifest: Path) -> list[Scan]:
    """Load and validate a fixed-support D/V manifest."""

    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest has no cases.")

    scans: list[Scan] = []
    for row in rows:
        paths = {field: _resolve(row[field], manifest) for field in REQUIRED_COLUMNS}
        anatomy_nii, anatomy = _load(paths["image"])
        arrays: dict[str, np.ndarray] = {"image": anatomy}
        for field in ("combined", "cord", "dorsal", "ventral"):
            candidate_nii, candidate = _labels(paths[field], description=field)
            _check_grid(candidate_nii, anatomy_nii, field=field)
            arrays[field] = candidate
        if not np.any(arrays["combined"] > 0):
            raise ValueError("Combined RootletSeg support is empty.")
        if not np.any(arrays["cord"] > 0):
            raise ValueError("Spinal-cord mask is empty.")
        if np.any((arrays["dorsal"] > 0) & (arrays["ventral"] > 0)) or not np.array_equal(
            arrays["dorsal"] + arrays["ventral"], arrays["combined"]
        ):
            raise ValueError("D/V outputs must exactly partition the fixed RootletSeg support.")
        scans.append(Scan(**arrays))
    return scans


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    values = image[np.isfinite(image)]
    lower, upper = np.percentile(values, (1, 99))
    return float(lower), float(upper if upper > lower else lower + 1.0)


def _support_slices(combined: np.ndarray, panels: int) -> list[int]:
    support = np.flatnonzero(np.any(combined > 0, axis=(0, 1)))
    if not support.size:
        raise ValueError("Cannot select slices from empty rootlet support.")
    positions = np.linspace(0, support.size - 1, panels)
    return [int(support[int(round(position))]) for position in positions]


def _crop_limits(scan: Scan, padding: int = 10) -> tuple[tuple[int, int], tuple[int, int]]:
    support = np.any((scan.combined > 0) | (scan.cord > 0), axis=2)
    x_values, y_values = np.nonzero(support)
    return (
        (max(0, int(x_values.min()) - padding), min(scan.image.shape[0], int(x_values.max()) + padding + 1)),
        (max(0, int(y_values.min()) - padding), min(scan.image.shape[1], int(y_values.max()) + padding + 1)),
    )


def _overlay(axis: plt.Axes, scan: Scan, z_index: int, *, split: bool, limits: tuple[tuple[int, int], tuple[int, int]]) -> None:
    lower, upper = _display_limits(scan.image)
    axis.imshow(scan.image[:, :, z_index].T, cmap="gray", origin="upper", vmin=lower, vmax=upper)
    if split:
        for mask, colour in ((scan.dorsal, DORSAL_COLOUR), (scan.ventral, VENTRAL_COLOUR)):
            values = mask[:, :, z_index].T > 0
            axis.imshow(
                np.ma.masked_where(~values, values),
                cmap=matplotlib.colors.ListedColormap([colour]),
                origin="upper",
                vmin=0,
                vmax=1,
            )
    else:
        values = scan.combined[:, :, z_index].T > 0
        axis.imshow(
            np.ma.masked_where(~values, values),
            cmap=matplotlib.colors.ListedColormap([ROOTLET_COLOUR]),
            origin="upper",
            vmin=0,
            vmax=1,
        )
    cord = scan.cord[:, :, z_index].T > 0
    if np.any(cord):
        axis.contour(cord.astype(np.uint8), levels=[0.5], colors=[CORD_COLOUR], linewidths=0.7)
    axis.set_xlim(limits[0])
    axis.set_ylim(limits[1][1], limits[1][0])
    axis.text(0.5, 1.01, "A", transform=axis.transAxes, ha="center", fontsize=7)
    axis.text(0.5, -0.04, "P", transform=axis.transAxes, ha="center", fontsize=7)
    axis.text(-0.03, 0.5, "L", transform=axis.transAxes, va="center", fontsize=7)
    axis.text(1.01, 0.5, "R", transform=axis.transAxes, va="center", fontsize=7)
    axis.axis("off")


def _representative_scan(scans: list[Scan]) -> Scan:
    """Return the median-fraction scan, breaking ties by its manifest position."""

    return sorted(enumerate(scans), key=lambda item: (item[1].dorsal_fraction, item[0]))[
        len(scans) // 2
    ][1]


def render_montage(
    scan: Scan,
    output: Path,
    *,
    panels: int,
    source_label: str,
    pipeline_label: str,
) -> None:
    """Render source-support and D/V overlays on the same fixed axial slices."""

    figure, axes = plt.subplots(2, panels, figsize=(2.45 * panels, 5.0), squeeze=False)
    limits = _crop_limits(scan)
    for column, z_index in enumerate(_support_slices(scan.combined, panels)):
        _overlay(axes[0, column], scan, z_index, split=False, limits=limits)
        _overlay(axes[1, column], scan, z_index, split=True, limits=limits)
        axes[0, column].set_title(f"axial slice {z_index}", fontsize=8)
    axes[0, 0].set_ylabel("Fixed RootletSeg support", fontsize=9)
    axes[1, 0].set_ylabel("D/V partition", fontsize=9)
    figure.suptitle(
        f"{source_label}\n"
        "Representative scan: median dorsal-support fraction\n"
        f"{pipeline_label}; qualitative QC only, not anatomical D/V accuracy",
        fontsize=12,
    )
    figure.legend(
        handles=[
            Patch(facecolor=ROOTLET_COLOUR, label="fixed RootletSeg support"),
            Patch(facecolor=DORSAL_COLOUR, label="predicted dorsal"),
            Patch(facecolor=VENTRAL_COLOUR, label="predicted ventral"),
            Patch(facecolor="none", edgecolor=CORD_COLOUR, label="spinal cord"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.85))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_gif(
    scan: Scan,
    output: Path,
    *,
    frames: int,
    source_label: str,
    pipeline_label: str,
) -> None:
    """Render a source-to-D/V axial sweep for the representative fixed-test scan."""

    limits = _crop_limits(scan)
    support = _support_slices(scan.combined, frames)
    images: list[Image.Image] = []
    for index, z_index in enumerate(support, start=1):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), squeeze=False)
        _overlay(axes[0, 0], scan, z_index, split=False, limits=limits)
        _overlay(axes[0, 1], scan, z_index, split=True, limits=limits)
        axes[0, 0].set_title("Fixed RootletSeg support", fontsize=9)
        axes[0, 1].set_title("D/V partition", fontsize=9)
        figure.suptitle(
            f"{source_label} · frame {index}/{frames} · axial slice {z_index}\n"
            f"{pipeline_label}; qualitative QC only, not anatomical D/V accuracy",
            fontsize=10,
        )
        figure.legend(
            handles=[
                Patch(facecolor=ROOTLET_COLOUR, label="fixed support"),
                Patch(facecolor=DORSAL_COLOUR, label="dorsal"),
                Patch(facecolor=VENTRAL_COLOUR, label="ventral"),
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.08, 1, 0.88))
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        buffer.seek(0)
        with Image.open(buffer) as frame:
            images.append(frame.convert("P", palette=Image.Palette.ADAPTIVE).copy())
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=180,
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_cohort_summary(
    scans: list[Scan], output: Path, *, source_label: str, pipeline_label: str
) -> None:
    """Render anonymous class-balance and exact-support QC panels."""

    ordered = sorted(scans, key=lambda scan: scan.dorsal_fraction)
    fractions = np.array([scan.dorsal_fraction for scan in ordered])
    support = np.array([scan.support_voxels for scan in ordered])
    output_support = np.array(
        [np.count_nonzero(scan.dorsal) + np.count_nonzero(scan.ventral) for scan in ordered]
    )
    labels = [f"Scan {index:02d}" for index in range(1, len(ordered) + 1)]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), gridspec_kw={"width_ratios": [1.28, 1]})
    y_values = np.arange(len(ordered))
    axes[0].barh(y_values, fractions * 100, color=DORSAL_COLOUR, label="dorsal")
    axes[0].barh(
        y_values,
        (1.0 - fractions) * 100,
        left=fractions * 100,
        color=VENTRAL_COLOUR,
        label="ventral",
    )
    mean_fraction = float(np.mean(fractions)) * 100
    axes[0].axvline(mean_fraction, color="black", linewidth=1.0)
    axes[0].text(
        mean_fraction + 1.0,
        len(ordered) - 0.4,
        f"mean {mean_fraction:.1f}% dorsal",
        fontsize=8,
        va="top",
    )
    axes[0].set(yticks=y_values, yticklabels=labels, xlim=(0, 100), xlabel="Fraction of fixed RootletSeg support (%)")
    axes[0].set_title("Class balance across fixed test scans", fontsize=11)
    axes[0].legend(loc="lower right", frameon=False)

    lower = max(0, int(support.min() * 0.95))
    upper = int(support.max() * 1.05)
    axes[1].plot([lower, upper], [lower, upper], color="black", linewidth=1.0, label="identity")
    axes[1].scatter(support, output_support, color=DORSAL_COLOUR, edgecolors="black", linewidths=0.35)
    axes[1].set(xlim=(lower, upper), ylim=(lower, upper), xlabel="RootletSeg support voxels", ylabel="Dorsal + ventral voxels")
    axes[1].set_title(f"Exact support preservation ({len(scans)}/{len(scans)} scans)", fontsize=11)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(loc="lower right", frameon=False)

    figure.suptitle(
        f"{source_label}\n{pipeline_label}: cohort QC", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    manifest: Path,
    output_directory: Path,
    *,
    panels: int = 6,
    frames: int = 24,
    dataset_label: str = "Dataset not specified",
    image_set_label: str = "Manifest cases",
    pipeline_label: str = "Combined support → D/V partition",
) -> list[Path]:
    """Write the review pack and a machine-readable aggregate summary."""

    if panels < 2 or frames < 2:
        raise ValueError("panels and frames must each be at least 2.")
    scans = read_manifest(manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    representative = _representative_scan(scans)
    source_label = f"Dataset: {dataset_label} | Images: {image_set_label}"
    outputs = [
        output_directory / "fixed-test_representative-montage.png",
        output_directory / "fixed-test_representative-sweep.gif",
        output_directory / "fixed-test_cohort-qc.png",
    ]
    render_montage(
        representative,
        outputs[0],
        panels=panels,
        source_label=source_label,
        pipeline_label=pipeline_label,
    )
    render_gif(
        representative,
        outputs[1],
        frames=frames,
        source_label=source_label,
        pipeline_label=pipeline_label,
    )
    render_cohort_summary(
        scans, outputs[2], source_label=source_label, pipeline_label=pipeline_label
    )
    summary = {
        "dataset": dataset_label,
        "image_set": image_set_label,
        "pipeline": pipeline_label,
        "cases": len(scans),
        "exact_partitions": len(scans),
        "representative_policy": "median dorsal-support fraction",
        "representative_dorsal_fraction": representative.dorsal_fraction,
        "dorsal_fraction_mean": float(np.mean([scan.dorsal_fraction for scan in scans])),
        "dorsal_fraction_range": [
            float(min(scan.dorsal_fraction for scan in scans)),
            float(max(scan.dorsal_fraction for scan in scans)),
        ],
        "interpretation": "QC only; no expert dorsal/ventral ground truth is available.",
    }
    summary_output = output_directory / "fixed-test_review-summary.json"
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    return [*outputs, summary_output]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--dataset-label", default="Dataset not specified")
    parser.add_argument("--image-set-label", default="Manifest cases")
    parser.add_argument("--pipeline-label", default="Combined support → D/V partition")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    for output in run(
        args.manifest,
        args.output_dir,
        panels=args.panels,
        frames=args.frames,
        dataset_label=args.dataset_label,
        image_set_label=args.image_set_label,
        pipeline_label=args.pipeline_label,
    ):
        print(output)


if __name__ == "__main__":
    main()
