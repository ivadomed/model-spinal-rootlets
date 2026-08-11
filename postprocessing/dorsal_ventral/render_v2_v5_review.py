#!/usr/bin/env python3
"""Render anonymized held-out speed/accuracy evidence for V2-V5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402


FULL_METHODS = ("V2", "V3", "V4", "V5 hybrid", "3-D classifier")
LEGACY_METHODS = ("V2", "V3", "V4")
DISPLAY_ROWS = (
    "Expert",
    "V2",
    "V3",
    "V4",
    "V5 gate",
    "V5 hybrid",
    "3-D classifier",
)
COLORS = {
    "dorsal": (0.94, 0.24, 0.24, 0.82),
    "ventral": (0.16, 0.55, 0.94, 0.82),
    "fallback": (1.0, 0.68, 0.08, 0.85),
    "error": (0.93, 0.0, 0.67, 0.92),
}
METHOD_LABELS = {
    "V5 gate": "V5\ngate",
    "V5 hybrid": "V5\nhybrid",
    "3-D classifier": "3-D\nclassifier",
}


def _data(path: Path) -> np.ndarray:
    image = nib.load(path)
    return np.rint(np.asanyarray(image.dataobj)).astype(np.uint8)


def _anatomy(path: Path) -> np.ndarray:
    image = nib.load(path)
    if tuple(nib.aff2axcodes(image.affine)) != ("R", "P", "I"):
        raise ValueError(f"Review input is not RPI: {path}")
    return np.asanyarray(image.dataobj).astype(np.float32)


def _case(case_directory: Path, alias: str) -> dict[str, Any]:
    directory = case_directory / alias.replace(" ", "_")
    return {
        "alias": alias,
        "anatomy": _anatomy(directory / "anatomy.nii.gz"),
        "rootlets": _data(directory / "rootlets.nii.gz"),
        "cord": _data(directory / "cord.nii.gz"),
        "Expert": _data(directory / "expert_dseg.nii.gz"),
        "V2": _data(directory / "V2_dseg.nii.gz"),
        "V3": _data(directory / "V3_dseg.nii.gz"),
        "V4": _data(directory / "V4_dseg.nii.gz"),
        "V5 gate": _data(directory / "V5_gate_dseg.nii.gz"),
        "V5 hybrid": _data(directory / "V5_hybrid_dseg.nii.gz"),
        "3-D classifier": _data(directory / "3-D_classifier_dseg.nii.gz"),
    }


def _limits(anatomy: np.ndarray, rootlets: np.ndarray) -> tuple[float, float]:
    support = rootlets > 0
    expanded = np.any(support, axis=2)
    values = anatomy[np.repeat(expanded[:, :, None], anatomy.shape[2], axis=2)]
    if not values.size:
        values = anatomy[np.isfinite(anatomy)]
    lower, upper = np.percentile(values, (1, 99))
    return float(lower), float(upper if upper > lower else lower + 1.0)


def _crop(rootlets: np.ndarray, padding: int = 10) -> tuple[slice, slice]:
    x, y, _z = np.nonzero(rootlets > 0)
    return (
        slice(
            max(0, int(x.min()) - padding),
            min(rootlets.shape[0], int(x.max()) + padding + 1),
        ),
        slice(
            max(0, int(y.min()) - padding),
            min(rootlets.shape[1], int(y.max()) + padding + 1),
        ),
    )


def _slice_scores(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    support_slices = np.flatnonzero(np.any(case["rootlets"] > 0, axis=(0, 1)))
    predictions = np.stack([case[name] for name in FULL_METHODS])
    expert = case["Expert"]
    errors = np.sum(
        (predictions != expert[None, ...])
        & (case["rootlets"][None, ...] > 0),
        axis=(0, 1, 2),
    )
    return support_slices, errors


def _selected_slices(case: dict[str, Any], panels: int) -> list[int]:
    support, errors = _slice_scores(case)
    selected: list[int] = []
    for group in np.array_split(support, panels):
        group_errors = errors[group]
        best = np.flatnonzero(group_errors == group_errors.max())
        selected.append(int(group[best[len(best) // 2]]))
    return selected


def _improvement_scores(case: dict[str, Any]) -> np.ndarray:
    """Count legacy errors corrected by V5 on each axial slice."""
    support = case["rootlets"] > 0
    expert = case["Expert"]
    legacy = np.stack([case[name] for name in LEGACY_METHODS])
    legacy_error = np.any(legacy != expert[None, ...], axis=0)
    v5_correct = case["V5 hybrid"] == expert
    return np.sum(support & legacy_error & v5_correct, axis=(0, 1))


def _improvement_slices(
    case: dict[str, Any], panels: int, minimum_gap: int = 6
) -> list[int]:
    scores = _improvement_scores(case)
    candidates = [
        int(index)
        for index in np.argsort(-scores, kind="stable")
        if scores[index] > 0
    ]
    selected: list[int] = []
    for index in candidates:
        if all(abs(index - previous) >= minimum_gap for previous in selected):
            selected.append(index)
        if len(selected) == panels:
            break
    if len(selected) < panels:
        for index in candidates:
            if index not in selected:
                selected.append(index)
            if len(selected) == panels:
                break
    if len(selected) < panels:
        raise ValueError(f"{case['alias']} has fewer than {panels} improvement slices.")
    return sorted(selected)


def _focus_slices(case: dict[str, Any]) -> list[tuple[int, str, int]]:
    """Choose one accepted and one abstained V5 decision, when available.

    Every selected slice contains a legacy error.  The first selection shows
    the deterministic route agreeing with the expert; the second shows the
    abstention route where both complete learned outputs agree with the expert.
    This makes the gate's role visible without treating an abstention as an
    incorrect D/V label.
    """

    support = case["rootlets"] > 0
    expert = case["Expert"]
    legacy_error = np.any(
        np.stack([case[name] for name in LEGACY_METHODS]) != expert[None, ...],
        axis=0,
    )
    gate = case["V5 gate"]
    hybrid = case["V5 hybrid"]
    classifier = case["3-D classifier"]
    routes = (
        (
            "V5 accepts:\ndeterministic label = expert",
            support & legacy_error & (gate > 0) & (gate == expert),
        ),
        (
            "V5 abstains:\nlearned outputs = expert",
            support
            & legacy_error
            & (gate == 0)
            & (hybrid == expert)
            & (classifier == expert),
        ),
    )
    selected: list[tuple[int, str, int]] = []
    for label, mask in routes:
        scores = np.sum(mask, axis=(0, 1))
        if np.any(scores):
            z_index = int(np.argmax(scores))
            selected.append((z_index, label, int(scores[z_index])))

    # A case can have no example of one route. Keep the visual honest and
    # still show its strongest complete V5 improvement instead of fabricating
    # a gate decision category.
    if not selected:
        scores = _improvement_scores(case)
        if not np.any(scores):
            raise ValueError(f"{case['alias']} has no V5 improvement slices.")
        z_index = int(np.argmax(scores))
        selected.append(
            (
                z_index,
                "V5 hybrid:\nlabel = expert",
                int(scores[z_index]),
            )
        )
    return selected


def _improvement_cases(
    summary: dict[str, Any], cases: list[dict[str, Any]], threshold: float = 0.01
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, case in zip(summary["cases"], cases):
        legacy_best = max(
            record["methods"][name]["voxel_accuracy"] for name in LEGACY_METHODS
        )
        v5 = record["methods"]["V5 hybrid"]["voxel_accuracy"]
        if v5 - legacy_best >= threshold:
            selected.append((record, case))
    return selected


def _overlay(
    axis: plt.Axes,
    anatomy: np.ndarray,
    prediction: np.ndarray,
    expert: np.ndarray,
    rootlets: np.ndarray,
    z_index: int,
    crop: tuple[slice, slice],
    limits: tuple[float, float],
    *,
    gate: bool,
) -> None:
    xs, ys = crop
    background = anatomy[xs, ys, z_index].T
    predicted = prediction[xs, ys, z_index].T
    truth = expert[xs, ys, z_index].T
    support = rootlets[xs, ys, z_index].T > 0
    axis.imshow(background, cmap="gray", origin="upper", vmin=limits[0], vmax=limits[1])
    for label, key in ((1, "dorsal"), (2, "ventral")):
        mask = predicted == label
        color = matplotlib.colors.ListedColormap([COLORS[key]])
        axis.imshow(
            np.ma.masked_where(~mask, mask),
            cmap=color,
            origin="upper",
            vmin=0,
            vmax=1,
        )
    if gate:
        fallback = support & (predicted == 0)
        color = matplotlib.colors.ListedColormap([COLORS["fallback"]])
        axis.imshow(
            np.ma.masked_where(~fallback, fallback),
            cmap=color,
            origin="upper",
            vmin=0,
            vmax=1,
        )
        error = support & (predicted > 0) & (predicted != truth)
    else:
        error = support & (predicted != truth)
    if np.any(error):
        color = matplotlib.colors.ListedColormap([COLORS["error"]])
        axis.imshow(
            np.ma.masked_where(~error, error),
            cmap=color,
            origin="upper",
            vmin=0,
            vmax=1,
        )
    axis.axis("off")


def render_case_montage(case: dict[str, Any], output: Path, panels: int) -> None:
    slices = _selected_slices(case, panels)
    crop = _crop(case["rootlets"])
    limits = _limits(case["anatomy"], case["rootlets"])
    figure, axes = plt.subplots(
        len(DISPLAY_ROWS),
        panels,
        figsize=(2.45 * panels, 2.1 * len(DISPLAY_ROWS)),
        squeeze=False,
    )
    for row, name in enumerate(DISPLAY_ROWS):
        for column, z_index in enumerate(slices):
            _overlay(
                axes[row, column],
                case["anatomy"],
                case[name],
                case["Expert"],
                case["rootlets"],
                z_index,
                crop,
                limits,
                gate=name == "V5 gate",
            )
            if row == 0:
                axes[row, column].set_title(f"slice {z_index}", fontsize=8)
            if column == 0:
                axes[row, column].text(
                    -0.08,
                    0.5,
                    name,
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
    figure.suptitle(
        f"{case['alias']} · frozen held-out test · RPI axial (top = anterior)\n"
        "same manual rootlet support for every method",
        fontsize=12,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS["dorsal"], label="dorsal"),
            Patch(facecolor=COLORS["ventral"], label="ventral"),
            Patch(facecolor=COLORS["fallback"], label="V5 abstained"),
            Patch(facecolor=COLORS["error"], label="expert disagreement"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.04, 0.05, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_improvement_grid(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    output: Path,
) -> None:
    selections = [(record, case, _focus_slices(case)) for record, case in selected]
    columns = sum(len(items) for _record, _case, items in selections)
    figure, axes = plt.subplots(
        len(DISPLAY_ROWS),
        columns,
        figsize=(2.4 * columns, 1.85 * len(DISPLAY_ROWS)),
        squeeze=False,
    )
    offset = 0
    for case_index, (record, case, slices) in enumerate(selections):
        crop = _crop(case["rootlets"])
        limits = _limits(case["anatomy"], case["rootlets"])
        for row, name in enumerate(DISPLAY_ROWS):
            for local_column, (z_index, reason, corrected_voxels) in enumerate(slices):
                column = offset + local_column
                _overlay(
                    axes[row, column],
                    case["anatomy"],
                    case[name],
                    case["Expert"],
                    case["rootlets"],
                    z_index,
                    crop,
                    limits,
                    gate=name == "V5 gate",
                )
                if row == 0:
                    axes[row, column].set_title(
                        f"{case['alias']} · slice {z_index}\n{reason}\n"
                        f"{corrected_voxels} legacy-error voxel(s) corrected",
                        fontsize=7,
                        fontweight="bold",
                        pad=8,
                    )
                if column == 0:
                    axes[row, column].text(
                        -0.08,
                        0.5,
                        name,
                        transform=axes[row, column].transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                    )
        if case_index:
            for row in range(len(DISPLAY_ROWS)):
                axes[row, offset].spines["left"].set_visible(True)
                axes[row, offset].spines["left"].set_color("#64748b")
                axes[row, offset].spines["left"].set_linewidth(1.5)
        offset += len(slices)
    figure.suptitle(
        "Failure-focused held-out comparison\n"
        "magenta = disagreement with expert · amber = V5 gate abstained, not a D/V error",
        fontsize=13,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS["dorsal"], label="dorsal"),
            Patch(facecolor=COLORS["ventral"], label="ventral"),
            Patch(facecolor=COLORS["fallback"], label="V5 gate abstained"),
            Patch(facecolor=COLORS["error"], label="expert disagreement"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.05, 0.05, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_cord_qc(cases: list[dict[str, Any]], output: Path) -> None:
    panels = 4
    figure, axes = plt.subplots(
        len(cases), panels, figsize=(2.55 * panels, 2.35 * len(cases)), squeeze=False
    )
    for row, case in enumerate(cases):
        support = np.flatnonzero(np.any(case["rootlets"] > 0, axis=(0, 1)))
        slices = [
            int(support[int(round(value))])
            for value in np.linspace(0, len(support) - 1, panels)
        ]
        crop = _crop(case["rootlets"])
        limits = _limits(case["anatomy"], case["rootlets"])
        for column, z_index in enumerate(slices):
            axis = axes[row, column]
            xs, ys = crop
            axis.imshow(
                case["anatomy"][xs, ys, z_index].T,
                cmap="gray",
                origin="upper",
                vmin=limits[0],
                vmax=limits[1],
            )
            rootlets = case["rootlets"][xs, ys, z_index].T > 0
            axis.imshow(
                np.ma.masked_where(~rootlets, rootlets),
                cmap=matplotlib.colors.ListedColormap([(1.0, 0.74, 0.0, 0.7)]),
                origin="upper",
                vmin=0,
                vmax=1,
            )
            cord = case["cord"][xs, ys, z_index].T > 0
            if np.any(cord):
                axis.contour(cord, levels=[0.5], colors=["#22d3ee"], linewidths=1.1)
            axis.set_title(f"slice {z_index}", fontsize=8)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    case["alias"],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
    figure.suptitle(
        "Shared V2–V4 cord input QC · RPI axial (top = anterior)\n"
        "yellow = manual rootlets · cyan = automatic SCT cord boundary",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.04, 0, 1, 0.92))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_metrics(summary: dict[str, Any], output: Path) -> None:
    methods = list(FULL_METHODS)
    labels = [METHOD_LABELS.get(name, name) for name in methods]
    metrics = summary["methods"]
    colors = ("#94a3b8", "#64748b", "#475569", "#2563eb")
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    x = np.arange(len(methods))
    width = 0.24
    for offset, key, title in (
        (-width, "voxel_balanced_accuracy", "Balanced accuracy"),
        (0.0, "dorsal_dice", "Dorsal Dice"),
        (width, "ventral_dice", "Ventral Dice"),
    ):
        axes[0].bar(
            x + offset,
            [metrics[name]["metrics"][key] * 100 for name in methods],
            width,
            label=title,
        )
    axes[0].set_title("Held-out voxel metrics")
    axes[0].set_ylabel("Percent")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 102)
    axes[0].legend(frameon=False, fontsize=8)

    component = [
        metrics[name]["metrics"]["simple_component_accuracy"] * 100
        for name in methods
    ]
    merged = [
        metrics[name]["metrics"]["merged_voxel_accuracy"] * 100
        for name in methods
    ]
    axes[1].bar(x - 0.18, component, 0.36, label="Simple components")
    axes[1].bar(x + 0.18, merged, 0.36, label="Merged-region voxels")
    axes[1].set_title("Component-level checks")
    axes[1].set_ylabel("Percent")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 102)
    axes[1].legend(frameon=False, fontsize=8)

    gate = summary["v5_gate"]
    gate_values = [
        gate["metrics"]["deterministic_voxel_accuracy"] * 100,
        gate["metrics"]["deterministic_voxel_coverage"] * 100,
        gate["metrics"]["selective_component_accuracy"] * 100,
        gate["metrics"]["component_coverage"] * 100,
    ]
    gate_labels = (
        "Voxel\naccuracy",
        "Voxel\ncoverage",
        "Component\naccuracy",
        "Component\ncoverage",
    )
    axes[2].bar(np.arange(4), gate_values, color=colors)
    axes[2].set_title("V5 deterministic gate (selective)")
    axes[2].set_ylabel("Percent")
    axes[2].set_xticks(np.arange(4), gate_labels)
    axes[2].set_ylim(0, 102)
    figure.suptitle(
        "Frozen held-out test · 3 cases · manual rootlet support\n"
        "V5 gate accuracy is selective; it is not a full-output score",
        fontsize=12,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_speed(summary: dict[str, Any], output: Path) -> None:
    method_names = (
        "V2",
        "V3",
        "V4",
        "V5 gate",
        "V5 hybrid",
        "3-D classifier",
    )
    values = [
        summary["methods"][name]["algorithm_timing"]["median_seconds"]
        for name in ("V2", "V3", "V4")
    ]
    values.append(summary["v5_gate"]["algorithm_timing"]["median_seconds"])
    values.append(summary["methods"]["V5 hybrid"]["seconds_per_case"])
    values.append(summary["methods"]["3-D classifier"]["seconds_per_case"])
    colors = (
        "#94a3b8",
        "#64748b",
        "#475569",
        "#f59e0b",
        "#2563eb",
        "#60a5fa",
    )
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    bars = axis.bar(
        [METHOD_LABELS.get(name, name) for name in method_names],
        values,
        color=colors,
    )
    axis.set_yscale("log")
    axis.set_ylabel("Seconds per case · log scale")
    axis.set_title(
        "D/V runtime after the rootlet label exists\n"
        "V2–V4 and V5 gate: CPU core; V5: locked GPU run + CPU combine"
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    cord = summary.get("cord_segmentation")
    if cord:
        axis.text(
            0.02,
            0.96,
            "V2–V4 also require a cord mask: "
            f"{cord['median_seconds']:.1f}s/case if generated now",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            color="#7c2d12",
        )
    benchmarks = summary.get("network_benchmarks", [])
    if len(benchmarks) > 1:
        recovery = benchmarks[1]
        figure.text(
            0.98,
            0.02,
            f"Recovery rerun: {recovery['seconds_per_case']:.2f}s/case\n"
            f"({recovery['label']})",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#1e3a8a",
        )
    axis.grid(axis="y", alpha=0.25, which="both")
    figure.tight_layout(rect=(0, 0.11 if len(benchmarks) > 1 else 0, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_per_case(summary: dict[str, Any], output: Path) -> None:
    methods = list(FULL_METHODS)
    matrix = np.asarray(
        [
            [case["methods"][name]["voxel_balanced_accuracy"] * 100 for name in methods]
            for case in summary["cases"]
        ]
    )
    figure, axis = plt.subplots(figsize=(7.6, 3.4))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(
        np.arange(len(methods)),
        [METHOD_LABELS.get(name, name) for name in methods],
    )
    axis.set_yticks(
        np.arange(len(summary["cases"])),
        [case["case"] for case in summary["cases"]],
    )
    axis.set_title("Balanced accuracy by held-out case")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.1f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 70 else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="Percent")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _gif_frame(case: dict[str, Any], z_index: int) -> Image.Image:
    crop = _crop(case["rootlets"])
    limits = _limits(case["anatomy"], case["rootlets"])
    figure, axes = plt.subplots(2, 4, figsize=(10.8, 5.9))
    for axis, name in zip(axes.flat, DISPLAY_ROWS):
        _overlay(
            axis,
            case["anatomy"],
            case[name],
            case["Expert"],
            case["rootlets"],
            z_index,
            crop,
            limits,
            gate=name == "V5 gate",
        )
        axis.set_title(name, fontsize=9)
    axes.flat[-1].axis("off")
    figure.suptitle(
        f"{case['alias']} · slice {z_index} · RPI top = anterior",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.canvas.draw()
    frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return frame


def render_sweep(case: dict[str, Any], output: Path, frames: int) -> None:
    support = np.flatnonzero(np.any(case["rootlets"] > 0, axis=(0, 1)))
    indices = [
        int(support[int(round(value))])
        for value in np.linspace(0, len(support) - 1, frames)
    ]
    images = [_gif_frame(case, index) for index in indices]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=280,
        loop=0,
        optimize=True,
    )


def _improvement_gif_frame(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    selections: list[tuple[int, str, int]],
) -> Image.Image:
    figure, axes = plt.subplots(
        len(selected),
        len(DISPLAY_ROWS),
        figsize=(17.0, 3.55 * len(selected)),
        squeeze=False,
    )
    for row, ((record, case), (z_index, reason, corrected_voxels)) in enumerate(
        zip(selected, selections)
    ):
        crop = _crop(case["rootlets"])
        limits = _limits(case["anatomy"], case["rootlets"])
        for column, name in enumerate(DISPLAY_ROWS):
            _overlay(
                axes[row, column],
                case["anatomy"],
                case[name],
                case["Expert"],
                case["rootlets"],
                z_index,
                crop,
                limits,
                gate=name == "V5 gate",
            )
            if row == 0:
                axes[row, column].set_title(name, fontsize=10, fontweight="bold")
        axes[row, 0].text(
            -0.10,
            0.5,
            f"{case['alias']}\nslice {z_index}\n{reason}\n"
            f"{corrected_voxels} legacy-error voxel(s) corrected",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    figure.suptitle(
        "V2–V5 failure-focused sweep · frozen held-out test\n"
        "amber in V5 gate = abstention; hybrid and classifier complete every rootlet voxel",
        fontsize=13,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS["dorsal"], label="dorsal"),
            Patch(facecolor=COLORS["ventral"], label="ventral"),
            Patch(facecolor=COLORS["fallback"], label="V5 gate abstained"),
            Patch(facecolor=COLORS["error"], label="expert disagreement"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.05, 0.04, 1, 0.94))
    figure.canvas.draw()
    frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return frame


def render_improvement_sweep(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    output: Path,
    frames: int,
) -> None:
    sequences = [_focus_slices(case) for _record, case in selected]
    images = [
        _improvement_gif_frame(
            selected,
            [sequence[frame % len(sequence)] for sequence in sequences],
        )
        for frame in range(frames)
    ]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=350,
        loop=0,
        optimize=True,
    )


def _flow_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    color: str,
    width: float = 0.25,
    height: float = 0.13,
) -> tuple[float, float, float, float]:
    x, y = xy
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.0,
        )
    )
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    return x, y, width, height


def _flow_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.3,
            color="#334155",
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )


def render_v5_decision_flow(summary: dict[str, Any], output: Path) -> None:
    """Render the deterministic gate, hybrid, and classifier-only paths."""

    gate = summary["v5_gate"]["metrics"]
    accepted = gate["deterministic_voxel_coverage"] * 100
    routed = 100 - accepted
    figure, axis = plt.subplots(figsize=(13.5, 5.6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")

    _flow_box(
        axis,
        (0.035, 0.43),
        "MRI + fixed\nRootletSeg support\n(level labels)",
        color="#e2e8f0",
    )
    _flow_box(
        axis,
        (0.355, 0.43),
        "V5 gate\n3-D components per level\nmean RPI Y split",
        color="#fef3c7",
        width=0.29,
    )
    _flow_box(
        axis,
        (0.71, 0.70),
        "V5 accepted labels\n(deterministic)",
        color="#dbeafe",
        width=0.24,
    )
    _flow_box(
        axis,
        (0.71, 0.20),
        "3-D nnU-Net\nlearned D/V labels",
        color="#dbeafe",
        width=0.24,
    )
    _flow_box(
        axis,
        (0.355, 0.08),
        "Classifier alone\nskips V5 and sends\nall support to 3-D nnU-Net",
        color="#e0f2fe",
        width=0.29,
        height=0.16,
    )
    _flow_arrow(axis, (0.285, 0.495), (0.355, 0.495))
    _flow_arrow(
        axis,
        (0.645, 0.54),
        (0.71, 0.765),
        f"accepts {accepted:.1f}%\nof held-out voxels",
    )
    _flow_arrow(
        axis,
        (0.645, 0.45),
        (0.71, 0.265),
        f"abstains on {routed:.1f}%\n→ fallback",
    )
    _flow_arrow(axis, (0.16, 0.43), (0.50, 0.24), "classifier-only path")
    _flow_arrow(axis, (0.645, 0.16), (0.71, 0.265))
    axis.text(
        0.83,
        0.52,
        "Hybrid output\n= V5 accepted labels +\nnetwork fallback labels",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#1e3a8a",
    )
    axis.text(
        0.50,
        0.94,
        "All final outputs are clipped to the fixed RootletSeg support: no rootlet voxels added or removed.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.50,
        0.86,
        "V5 gate is selective confidence/QC; hybrid and classifier alone are complete D/V outputs.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    summary_path: Path,
    case_directory: Path,
    output_directory: Path,
    *,
    panels: int,
    frames: int,
) -> list[Path]:
    summary = json.loads(summary_path.read_text())
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = [_case(case_directory, record["case"]) for record in summary["cases"]]
    outputs: list[Path] = []
    for case in cases:
        output = output_directory / (
            f"{case['alias'].replace(' ', '_')}_v2-v5_montage.png"
        )
        render_case_montage(case, output, panels)
        outputs.append(output)
    metrics = output_directory / "v2-v5_accuracy.png"
    speed = output_directory / "v2-v5_speed.png"
    per_case = output_directory / "v2-v5_per_case.png"
    cord_qc = output_directory / "v2-v4_cord_qc.png"
    render_metrics(summary, metrics)
    render_speed(summary, speed)
    render_per_case(summary, per_case)
    render_cord_qc(cases, cord_qc)
    outputs.extend((metrics, speed, per_case, cord_qc))
    decision_flow = output_directory / "v5_decision_paths.png"
    render_v5_decision_flow(summary, decision_flow)
    outputs.append(decision_flow)
    representative_index = sorted(
        range(len(summary["cases"])),
        key=lambda index: (
            summary["cases"][index]["methods"]["V5 hybrid"]["voxel_accuracy"],
            index,
        ),
    )[len(cases) // 2]
    sweep = output_directory / "v2-v5_representative_sweep.gif"
    render_sweep(cases[representative_index], sweep, frames)
    outputs.append(sweep)
    improvement = _improvement_cases(summary, cases)
    if improvement:
        improvement_grid = output_directory / "v2-v5_improvement_cases_grid.png"
        improvement_sweep = output_directory / "v2-v5_improvement_cases_sweep.gif"
        render_improvement_grid(improvement, improvement_grid)
        render_improvement_sweep(improvement, improvement_sweep, frames)
        outputs.extend((improvement_grid, improvement_sweep))
    manifest = {
        "schema": "rootlet-dv-v2-v5-review-v1",
        "split": summary["split"],
        "case_count": len(cases),
        "orientation": "RPI axial; display top is anterior",
        "representative_policy": "median V5 hybrid held-out voxel accuracy",
        "representative_case": cases[representative_index]["alias"],
        "improvement_policy": (
            "V5 hybrid voxel accuracy at least 1 percentage point above "
            "the best of V2-V4"
        ),
        "failure_focus_policy": (
            "Each shown slice contains a V2-V4 error and is selected to show "
            "either a correct V5 gate decision or a V5 gate abstention resolved "
            "by both learned outputs."
        ),
        "improvement_cases": [case["alias"] for _record, case in improvement],
        "files": [path.name for path in outputs],
    }
    manifest_path = output_directory / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs.append(manifest_path)
    return outputs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--case-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--frames", type=int, default=24)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    for output in run(
        args.summary_json,
        args.case_directory,
        args.output_directory,
        panels=args.panels,
        frames=args.frames,
    ):
        print(output)


if __name__ == "__main__":
    main()
