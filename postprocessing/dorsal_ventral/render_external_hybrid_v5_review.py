#!/usr/bin/env python3
"""Render dataset-labelled qualitative QC for unlabelled hybrid V5 cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from postprocessing.dorsal_ventral.render_hybrid_v5_review import (  # noqa: E402
    DORSAL,
    VENTRAL,
    _class_overlay,
    _crop,
    _draw,
    _slices,
    _window,
)


ROOTLETS = (1.00, 0.72, 0.08, 0.80)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _load_case(record: dict[str, Any]) -> dict[str, np.ndarray]:
    images = []
    for field in ("rpi_image", "rpi_rootlets", "class_map"):
        image = nib.load(record[field])
        images.append(image)
    reference = images[0]
    for candidate in images[1:]:
        if reference.shape != candidate.shape or not np.allclose(
            reference.affine, candidate.affine, atol=1e-4
        ):
            raise ValueError(f"External QC grid differs for {record['case_id']}.")
    if tuple(nib.aff2axcodes(reference.affine)) != ("R", "P", "I"):
        raise ValueError(f"External QC case is not RPI: {record['case_id']}.")
    return {
        "anatomy": np.asanyarray(images[0].dataobj),
        "rootlets": np.rint(np.asanyarray(images[1].dataobj)).astype(np.int16),
        "classes": np.rint(np.asanyarray(images[2].dataobj)).astype(np.uint8),
    }


def _rootlet_overlay(rootlets: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*rootlets.shape, 4), dtype=float)
    overlay[rootlets > 0] = ROOTLETS
    return overlay


def render_montage(
    data: dict[str, np.ndarray], output: Path, dataset: str, panels: int
) -> None:
    support = data["rootlets"] > 0
    crop = _crop(support)
    window = _window(data["anatomy"], support)
    slices = _slices(support, panels)
    overlays = (_rootlet_overlay(data["rootlets"]), _class_overlay(data["classes"]))
    figure, axes = plt.subplots(2, panels, figsize=(2.3 * panels, 4.8), squeeze=False)
    for row, (overlay, label) in enumerate(zip(overlays, ("RootletSeg", "Hybrid V5"))):
        for column, z_index in enumerate(slices):
            _draw(axes[row, column], data["anatomy"], overlay, z_index, crop, window)
            if row == 0:
                axes[row, column].set_title(f"slice {z_index}", fontsize=8)
        axes[row, 0].set_ylabel(label, fontsize=9)
    figure.suptitle(
        f"{dataset} · representative scan by median V5 coverage\n"
        "RPI axial: top = anterior · qualitative QC only; no expert D/V labels",
        fontsize=11,
    )
    figure.legend(
        handles=[
            Patch(facecolor=ROOTLETS, label="RootletSeg support"),
            Patch(facecolor=DORSAL, label="predicted dorsal"),
            Patch(facecolor=VENTRAL, label="predicted ventral"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.87))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_sweep(
    data: dict[str, np.ndarray], output: Path, dataset: str, frames: int
) -> None:
    support = data["rootlets"] > 0
    crop = _crop(support)
    window = _window(data["anatomy"], support)
    rootlet_overlay = _rootlet_overlay(data["rootlets"])
    class_overlay = _class_overlay(data["classes"])
    images: list[Image.Image] = []
    for index, z_index in enumerate(_slices(support, frames), start=1):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
        for axis, overlay, title in zip(
            axes,
            (rootlet_overlay, class_overlay),
            ("RootletSeg support", "Hybrid V5 D/V"),
        ):
            _draw(axis, data["anatomy"], overlay, z_index, crop, window)
            axis.set_title(title, fontsize=9)
        figure.suptitle(
            f"{dataset} · frame {index}/{frames} · slice {z_index}\n"
            "top = anterior · qualitative QC only",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.86))
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


def render_cohort(records: list[dict[str, Any]], output: Path) -> None:
    datasets = sorted(set(record["dataset"] for record in records))
    values = {
        dataset: [record for record in records if record["dataset"] == dataset]
        for dataset in datasets
    }
    labels = [dataset.replace("/data-multi-subject", "") for dataset in datasets]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    for axis, field, title in (
        (axes[0], "v5_level_coverage", "Levels handled deterministically"),
        (axes[1], "fallback_voxel_fraction", "Rootlet voxels routed to fallback"),
        (axes[2], "dorsal_fraction", "Predicted dorsal fraction"),
    ):
        series = [[record[field] * 100 for record in values[dataset]] for dataset in datasets]
        axis.boxplot(series, tick_labels=labels, showmeans=True)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percent")
        axis.set_title(title, fontsize=10)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Hybrid V5 external QC · {len(records)} scans · exact support preserved\n"
        "No expert dorsal/ventral ground truth: distributions are not accuracy",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(summary_json: Path, output_directory: Path, *, panels: int, frames: int) -> list[Path]:
    summary = json.loads(summary_json.read_text())
    records = summary["cases"]
    if summary["exact_support_partitions"] != len(records):
        raise ValueError("Refusing QC: not every external output preserves support.")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    datasets = sorted(set(record["dataset"] for record in records))
    for dataset in datasets:
        subset = [record for record in records if record["dataset"] == dataset]
        representative = sorted(
            enumerate(subset), key=lambda item: (item[1]["v5_level_coverage"], item[0])
        )[len(subset) // 2][1]
        data = _load_case(representative)
        slug = _slug(dataset)
        montage = output_directory / f"{slug}_representative_montage.png"
        sweep = output_directory / f"{slug}_representative_sweep.gif"
        render_montage(data, montage, dataset, panels)
        render_sweep(data, sweep, dataset, frames)
        outputs.extend((montage, sweep))

    cohort = output_directory / "external_cohort_qc.png"
    render_cohort(records, cohort)
    outputs.append(cohort)
    table = output_directory / "external_dataset_summary.csv"
    with table.open("w", newline="") as stream:
        fields = (
            "dataset",
            "cases",
            "v5_level_coverage_mean",
            "v5_level_coverage_range",
            "fallback_voxel_fraction_mean",
            "dorsal_fraction_mean",
            "exact_support_partitions",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for dataset in datasets:
            subset = [record for record in records if record["dataset"] == dataset]
            coverage = [record["v5_level_coverage"] for record in subset]
            writer.writerow(
                {
                    "dataset": dataset,
                    "cases": len(subset),
                    "v5_level_coverage_mean": float(np.mean(coverage)),
                    "v5_level_coverage_range": f"{min(coverage):.6f}–{max(coverage):.6f}",
                    "fallback_voxel_fraction_mean": float(
                        np.mean([record["fallback_voxel_fraction"] for record in subset])
                    ),
                    "dorsal_fraction_mean": float(
                        np.mean([record["dorsal_fraction"] for record in subset])
                    ),
                    "exact_support_partitions": sum(
                        record["support_preserved"] for record in subset
                    ),
                }
            )
    outputs.append(table)
    review = {
        "schema": "rootlet-dv-hybrid-v5-external-review-v1",
        "datasets": datasets,
        "case_count": len(records),
        "representative_policy": "median V5 deterministic level coverage within each dataset",
        "interpretation": "Qualitative and engineering QC only; no expert D/V ground truth.",
        "files": [path.name for path in outputs],
    }
    review_path = output_directory / "review_manifest.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    outputs.append(review_path)
    return outputs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--frames", type=int, default=24)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    for output in run(
        args.summary_json,
        args.output_directory,
        panels=args.panels,
        frames=args.frames,
    ):
        print(output)


if __name__ == "__main__":
    main()
