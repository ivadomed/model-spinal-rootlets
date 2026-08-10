#!/usr/bin/env python3
"""Render anonymized held-out evidence for the hybrid V5 D/V pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

from postprocessing.dorsal_ventral.cluster_mean_v5 import split_cluster_mean_v5  # noqa: E402


DORSAL = (0.90, 0.15, 0.15, 0.78)
VENTRAL = (0.05, 0.55, 0.95, 0.78)
FALLBACK = (1.00, 0.72, 0.08, 0.85)
ERROR = (0.95, 0.10, 0.75, 0.95)


def _load(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    return image, np.asanyarray(image.dataobj)


def _check_grid(reference: nib.Nifti1Image, candidate: nib.Nifti1Image) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference.affine, candidate.affine, atol=1e-4
    ):
        raise ValueError("Held-out evidence inputs do not share one grid.")


def _crop(support: np.ndarray, padding: int = 12) -> tuple[slice, slice]:
    x, y, _z = np.nonzero(support)
    if not len(x):
        raise ValueError("Cannot render empty rootlet support.")
    return (
        slice(max(0, int(x.min()) - padding), min(support.shape[0], int(x.max()) + padding + 1)),
        slice(max(0, int(y.min()) - padding), min(support.shape[1], int(y.max()) + padding + 1)),
    )


def _window(anatomy: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    expanded = np.any(support, axis=2)
    x, y = np.nonzero(expanded)
    values = anatomy[
        max(0, int(x.min()) - 20) : min(anatomy.shape[0], int(x.max()) + 21),
        max(0, int(y.min()) - 20) : min(anatomy.shape[1], int(y.max()) + 21),
        :,
    ]
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, (1, 99))
    return float(low), float(high if high > low else low + 1.0)


def _slices(support: np.ndarray, count: int) -> list[int]:
    available = np.flatnonzero(np.any(support, axis=(0, 1)))
    positions = np.linspace(0, len(available) - 1, count)
    return [int(available[int(round(position))]) for position in positions]


def _class_overlay(classes: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*classes.shape, 4), dtype=float)
    overlay[classes == 1] = DORSAL
    overlay[classes == 2] = VENTRAL
    return overlay


def _draw(
    axis: plt.Axes,
    anatomy: np.ndarray,
    overlay: np.ndarray,
    z_index: int,
    crop: tuple[slice, slice],
    window: tuple[float, float],
) -> None:
    x, y = crop
    axis.imshow(
        anatomy[x, y, z_index].T,
        cmap="gray",
        origin="upper",
        vmin=window[0],
        vmax=window[1],
    )
    axis.imshow(overlay[x, y, z_index].transpose(1, 0, 2), origin="upper")
    axis.set_xticks([])
    axis.set_yticks([])


def _case_data(
    dataset_directory: Path,
    hybrid_directory: Path,
    case_id: str,
    split: str,
) -> dict[str, Any]:
    image_folder = "imagesTs" if split == "test" else "imagesTr"
    label_folder = "labelsTs" if split == "test" else "labelsTr"
    anatomy_nii, anatomy = _load(dataset_directory / image_folder / f"{case_id}_0000.nii.gz")
    rootlet_nii, rootlets = _load(dataset_directory / image_folder / f"{case_id}_0001.nii.gz")
    truth_nii, truth = _load(dataset_directory / label_folder / f"{case_id}.nii.gz")
    hybrid_nii, hybrid = _load(hybrid_directory / f"{case_id}_desc-hybridV5_dseg.nii.gz")
    for candidate in (rootlet_nii, truth_nii, hybrid_nii):
        _check_grid(anatomy_nii, candidate)
    rootlets = np.rint(rootlets).astype(np.int16)
    truth = np.rint(truth).astype(np.uint8)
    hybrid = np.rint(hybrid).astype(np.uint8)
    v5 = split_cluster_mean_v5(
        rootlets, float(nib.affines.voxel_sizes(rootlet_nii.affine)[1])
    )
    return {
        "anatomy": anatomy,
        "rootlets": rootlets,
        "truth": truth,
        "hybrid": hybrid,
        "fallback": v5.fallback > 0,
    }


def render_case_montage(
    data: dict[str, Any], output: Path, alias: str, panels: int
) -> None:
    support = data["rootlets"] > 0
    crop = _crop(support)
    window = _window(data["anatomy"], support)
    slices = _slices(support, panels)
    truth_overlay = _class_overlay(data["truth"])
    hybrid_overlay = _class_overlay(data["hybrid"])
    fallback_overlay = np.zeros((*support.shape, 4), dtype=float)
    fallback_overlay[data["fallback"]] = FALLBACK
    error_overlay = np.zeros((*support.shape, 4), dtype=float)
    error_overlay[support & (data["truth"] != data["hybrid"])] = ERROR
    overlays = (truth_overlay, hybrid_overlay, fallback_overlay, error_overlay)
    labels = ("Expert D/V", "Hybrid V5", "Routed to fallback", "Errors")

    figure, axes = plt.subplots(4, panels, figsize=(2.25 * panels, 8.0), squeeze=False)
    for row, (overlay, label) in enumerate(zip(overlays, labels)):
        for column, z_index in enumerate(slices):
            _draw(axes[row, column], data["anatomy"], overlay, z_index, crop, window)
            if row == 0:
                axes[row, column].set_title(f"slice {z_index}", fontsize=8)
        axes[row, 0].set_ylabel(label, fontsize=9)
    figure.suptitle(
        f"Held-out {alias} · RPI axial views\n"
        "top = anterior; ventral = blue; dorsal = red",
        fontsize=12,
    )
    figure.legend(
        handles=[
            Patch(facecolor=DORSAL, label="dorsal"),
            Patch(facecolor=VENTRAL, label="ventral"),
            Patch(facecolor=FALLBACK, label="fallback region"),
            Patch(facecolor=ERROR, label="expert disagreement"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.91))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_sweep(
    data: dict[str, Any], output: Path, alias: str, frames: int
) -> None:
    support = data["rootlets"] > 0
    crop = _crop(support)
    window = _window(data["anatomy"], support)
    truth_overlay = _class_overlay(data["truth"])
    hybrid_overlay = _class_overlay(data["hybrid"])
    error_overlay = np.zeros((*support.shape, 4), dtype=float)
    error_overlay[support & (data["truth"] != data["hybrid"])] = ERROR
    images: list[Image.Image] = []
    for index, z_index in enumerate(_slices(support, frames), start=1):
        figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.4))
        for axis, overlay, title in zip(
            axes,
            (truth_overlay, hybrid_overlay, error_overlay),
            ("Expert", "Hybrid V5", "Disagreement"),
        ):
            _draw(axis, data["anatomy"], overlay, z_index, crop, window)
            axis.set_title(title, fontsize=9)
        figure.suptitle(
            f"Held-out {alias} · frame {index}/{frames} · axial slice {z_index}\n"
            "RPI: top = anterior",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.88))
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


def render_metrics(evaluation: dict[str, Any], output: Path) -> None:
    metrics = (
        ("voxel_accuracy", "Voxel accuracy"),
        ("voxel_balanced_accuracy", "Balanced accuracy"),
        ("dorsal_dice", "Dorsal Dice"),
        ("ventral_dice", "Ventral Dice"),
        ("simple_component_accuracy", "Component accuracy"),
        ("merged_voxel_accuracy", "Merged-region accuracy"),
    )
    model = [evaluation["model_only"].get(key) for key, _label in metrics]
    hybrid = [evaluation["hybrid"].get(key) for key, _label in metrics]
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(x - 0.2, [value or 0 for value in model], 0.4, label="fallback alone", color="#9ca3af")
    axis.bar(x + 0.2, [value or 0 for value in hybrid], 0.4, label="V5 + fallback", color="#2563eb")
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("Score")
    axis.set_xticks(x, [label for _key, label in metrics], rotation=18, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    axis.set_title(
        f"Held-out test ({evaluation['case_count']} cases) · frozen before training",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_flow(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.0, 3.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3.5)
    axis.axis("off")
    boxes = (
        (0.2, "RootletSeg mask\n(level labels)"),
        (2.6, "RPI component count\n+ mean y"),
        (5.1, "Clear 3–4 clusters\nV5 sorts by y"),
        (5.1, "Merged / ambiguous\nlevel"),
        (7.8, "Tiny D/V network\nonly here"),
        (10.2, "Exact same support\nD/V output"),
    )
    positions = ((0.2, 1.25), (2.6, 1.25), (5.1, 2.25), (5.1, 0.25), (7.8, 0.25), (10.2, 1.25))
    for (unused, text), (x, y) in zip(boxes, positions):
        del unused
        patch = FancyBboxPatch(
            (x, y), 1.8, 0.8, boxstyle="round,pad=0.08", facecolor="#eef2ff", edgecolor="#4f46e5"
        )
        axis.add_patch(patch)
        axis.text(x + 0.9, y + 0.4, text, ha="center", va="center", fontsize=9)
    arrows = (
        ((2.0, 1.65), (2.6, 1.65)),
        ((4.4, 1.65), (5.1, 2.65)),
        ((4.4, 1.55), (5.1, 0.65)),
        ((6.9, 0.65), (7.8, 0.65)),
        ((6.9, 2.65), (10.2, 1.75)),
        ((9.6, 0.65), (10.2, 1.55)),
    )
    for start, stop in arrows:
        axis.annotate("", xy=stop, xytext=start, arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.4})
    axis.text(5.95, 3.2, "deterministic branch", ha="center", fontsize=8, color="#475569")
    axis.text(7.0, 0.05, "learned fallback", ha="center", fontsize=8, color="#475569")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    dataset_directory: Path,
    hybrid_directory: Path,
    evaluation_json: Path,
    output_directory: Path,
    *,
    panels: int = 5,
    frames: int = 24,
) -> list[Path]:
    evaluation = json.loads(evaluation_json.read_text())
    split = evaluation["split"]
    output_directory.mkdir(parents=True, exist_ok=True)
    case_records = evaluation["cases"]
    outputs: list[Path] = []
    table_path = output_directory / "heldout_per_case.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("case", "hybrid_voxel_accuracy", "hybrid_balanced_accuracy", "v5_level_coverage"),
        )
        writer.writeheader()
        for index, record in enumerate(case_records):
            alias = f"Case {chr(ord('A') + index)}"
            writer.writerow(
                {
                    "case": alias,
                    "hybrid_voxel_accuracy": record["hybrid"]["voxel_accuracy"],
                    "hybrid_balanced_accuracy": record["hybrid"]["voxel_balanced_accuracy"],
                    "v5_level_coverage": record["v5_level_coverage"],
                }
            )
            data = _case_data(dataset_directory, hybrid_directory, record["case_id"], split)
            montage = output_directory / f"heldout_case_{chr(ord('A') + index)}_montage.png"
            render_case_montage(data, montage, alias, panels)
            outputs.append(montage)

    representative_index = sorted(
        range(len(case_records)),
        key=lambda index: (case_records[index]["hybrid"]["voxel_accuracy"], index),
    )[len(case_records) // 2]
    representative = case_records[representative_index]
    alias = f"Case {chr(ord('A') + representative_index)}"
    data = _case_data(dataset_directory, hybrid_directory, representative["case_id"], split)
    sweep = output_directory / "heldout_median_accuracy_sweep.gif"
    render_sweep(data, sweep, alias, frames)
    metrics = output_directory / "heldout_metrics.png"
    flow = output_directory / "hybrid_v5_flow.png"
    render_metrics(evaluation, metrics)
    render_flow(flow)
    outputs.extend((sweep, metrics, flow, table_path))
    manifest = {
        "schema": "rootlet-dv-hybrid-v5-review-v1",
        "split": split,
        "case_count": len(case_records),
        "case_aliases": [f"Case {chr(ord('A') + index)}" for index in range(len(case_records))],
        "representative_policy": "median held-out hybrid voxel accuracy",
        "representative_alias": alias,
        "orientation": "RPI axial; display top is anterior",
        "files": [path.name for path in outputs],
    }
    manifest_path = output_directory / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs.append(manifest_path)
    return outputs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--hybrid-directory", required=True, type=Path)
    parser.add_argument("--evaluation-json", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--panels", type=int, default=5)
    parser.add_argument("--frames", type=int, default=24)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    for output in run(
        args.dataset_directory,
        args.hybrid_directory,
        args.evaluation_json,
        args.output_directory,
        panels=args.panels,
        frames=args.frames,
    ):
        print(output)


if __name__ == "__main__":
    main()
