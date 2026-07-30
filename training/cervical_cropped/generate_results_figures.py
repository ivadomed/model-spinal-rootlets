#!/usr/bin/env python3
"""Generate quantitative and qualitative figures for cropped-rootlets results.

The quantitative figures only require ``metrics_per_case.csv``. Qualitative
figures are additionally generated when the image, reference, and prediction
directories are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


LEVELS = {
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    8: "C8",
    9: "T1",
}
CONTRASTS = ["T2w", "INV1", "INV2", "UNIT1"]
COLORS = {
    "T2w": "#009E73",
    "INV1": "#D55E00",
    "INV2": "#0072B2",
    "UNIT1": "#CC79A7",
}
GT_COLOR = "#00BFC4"
PRED_COLOR = "#F8766D"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate test Dice charts and raw/reference/prediction montages."
    )
    parser.add_argument("--metrics-per-case", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument(
        "--model-label",
        default="Cropped RPI model",
        help="Model name used in quantitative figure titles.",
    )
    parser.add_argument("--dpi", default=300, type=int)
    return parser


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def point_offsets(n: int, width: float = 0.18) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def add_box_and_points(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    width: float,
    point_width: float,
) -> None:
    if not len(values):
        return
    box = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"edgecolor": color, "linewidth": 1.2},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
    )
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_alpha(0.28)
    ax.scatter(
        position + point_offsets(len(values), point_width),
        values,
        s=28,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        [position],
        [float(np.mean(values))],
        marker="D",
        s=28,
        facecolor="white",
        edgecolor=color,
        linewidth=1.2,
        zorder=4,
    )


def style_dice_axis(ax: plt.Axes) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_dice_by_contrast(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    model_label: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
    panels = [
        ("macro_level_dice", "Macro level Dice"),
        ("binary_dice", "Binary foreground Dice"),
    ]
    for ax, (column, title) in zip(axes, panels):
        for index, contrast in enumerate(CONTRASTS):
            values = finite_values(df.loc[df["group"] == contrast, column])
            add_box_and_points(ax, values, index, COLORS[contrast], 0.55, 0.09)
            ax.text(index, 1.01, f"n={len(values)}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(CONTRASTS)), CONTRASTS)
        ax.set_xlabel("MRI contrast")
        ax.set_title(title, fontweight="bold")
        style_dice_axis(ax)
    axes[0].set_ylabel("Dice coefficient")
    fig.suptitle(f"{model_label}: held-out test performance", fontweight="bold")
    handles = [
        Patch(facecolor="white", edgecolor="#555555", label="Median and IQR"),
        plt.Line2D([], [], marker="o", linestyle="", color="#555555", label="Individual image"),
        plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                   markeredgecolor="#555555", label="Mean"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    path = output_dir / "test_dice_by_contrast.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def level_is_present(levels: object, level: str) -> bool:
    """Return whether a semicolon-delimited level inventory contains ``level``."""
    if pd.isna(levels):
        return False
    return level in {item.strip() for item in str(levels).split(";") if item.strip()}


def plot_dice_by_level(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    model_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    offsets = dict(zip(CONTRASTS, np.linspace(-0.30, 0.30, len(CONTRASTS))))
    width = 0.17
    for level_index, level_name in enumerate(LEVELS.values()):
        column = f"dice_{level_name}"
        for contrast in CONTRASTS:
            values = finite_values(df.loc[df["group"] == contrast, column])
            add_box_and_points(
                ax,
                values,
                level_index + offsets[contrast],
                COLORS[contrast],
                width,
                0.025,
            )
    ax.set_xticks(range(len(LEVELS)), LEVELS.values())
    ax.set_xlabel("Spinal level")
    ax.set_ylabel("Dice coefficient")
    ax.set_title(f"{model_label}: test Dice by spinal level and contrast", fontweight="bold")
    style_dice_axis(ax)
    ax.axvline(6.5, color="#777777", linestyle=":", linewidth=1)
    t1_reference_count = int(df["gt_levels"].map(lambda levels: level_is_present(levels, "T1")).sum())
    t1_prediction_count = int(df["pred_levels"].map(lambda levels: level_is_present(levels, "T1")).sum())
    ax.text(
        7,
        1.01,
        f"T1 reference: n={t1_reference_count}; predicted: n={t1_prediction_count}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.legend(
        handles=[Patch(facecolor=COLORS[name], alpha=0.55, label=name) for name in CONTRASTS],
        title="Contrast",
        loc="lower left",
        ncol=4,
        frameon=False,
    )
    path = output_dir / "test_dice_by_spinal_level.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def find_case_file(directory: Path, case_id: str, is_image: bool) -> Path:
    candidates = [directory / f"{case_id}_0000.nii.gz", directory / f"{case_id}.nii.gz"] if is_image else [
        directory / f"{case_id}.nii.gz"
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No file found for {case_id} in {directory}")


def load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("Qualitative figures require nibabel.") from error
    image = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(image.dataobj), image.affine


def load_case(
    case_id: str,
    images_dir: Path,
    labels_dir: Path,
    predictions_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, image_affine = load_canonical(find_case_file(images_dir, case_id, is_image=True))
    reference, reference_affine = load_canonical(find_case_file(labels_dir, case_id, is_image=False))
    prediction, prediction_affine = load_canonical(find_case_file(predictions_dir, case_id, is_image=False))
    if image.shape != reference.shape or image.shape != prediction.shape:
        raise ValueError(
            f"Grid shape mismatch for {case_id}: image={image.shape}, "
            f"reference={reference.shape}, prediction={prediction.shape}"
        )
    if not np.allclose(image_affine, reference_affine, atol=1e-4) or not np.allclose(
        image_affine, prediction_affine, atol=1e-4
    ):
        raise ValueError(f"Affine mismatch for {case_id}")
    return image.astype(np.float32), np.rint(reference).astype(np.int16), np.rint(prediction).astype(np.int16)


def axial_slice(volume: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(volume[:, :, index])


def select_slice(reference: np.ndarray, prediction: np.ndarray, label: int) -> int:
    support = np.sum(reference == label, axis=(0, 1)) + np.sum(prediction == label, axis=(0, 1))
    if not np.any(support):
        raise ValueError(f"Neither reference nor prediction contains label {label}")
    return int(np.argmax(support))


def select_representative_level(row: pd.Series) -> tuple[int, str]:
    macro = float(row["macro_level_dice"])
    candidates: list[tuple[float, int, str]] = []
    for label, name in LEVELS.items():
        value = pd.to_numeric(pd.Series([row[f"dice_{name}"]]), errors="coerce").iloc[0]
        if math.isfinite(float(value)):
            candidates.append((abs(float(value) - macro), label, name))
    if not candidates:
        raise ValueError(f"No valid level Dice for {row['case_id']}")
    _, label, name = min(candidates)
    return label, name


def square_crop(mask: np.ndarray, padding: int = 12) -> tuple[slice, slice]:
    coordinates = np.argwhere(mask)
    height, width = mask.shape
    if not len(coordinates):
        return slice(0, height), slice(0, width)
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    center_y = (int(y_min) + int(y_max)) // 2
    center_x = (int(x_min) + int(x_max)) // 2
    side = max(int(y_max - y_min + 1), int(x_max - x_min + 1)) + 2 * padding
    side = min(max(side, min(height, width) // 2), height, width)
    y0 = min(max(center_y - side // 2, 0), height - side)
    x0 = min(max(center_x - side // 2, 0), width - side)
    return slice(y0, y0 + side), slice(x0, x0 + side)


def intensity_limits(image: np.ndarray, crop: tuple[slice, slice]) -> tuple[float, float]:
    values = image[crop]
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    nonzero = values[values != 0]
    if len(nonzero) >= 20:
        values = nonzero
    low, high = np.percentile(values, [1, 99])
    if not high > low:
        high = low + 1.0
    return float(low), float(high)


def draw_anatomy_panel(
    ax: plt.Axes,
    image: np.ndarray,
    crop: tuple[slice, slice],
    overlay: np.ndarray | None = None,
    overlay_color: str | None = None,
    empty_text: str | None = None,
) -> None:
    low, high = intensity_limits(image, crop)
    ax.imshow(image[crop], cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    if overlay is not None and np.any(overlay[crop]):
        color_map = ListedColormap([(0, 0, 0, 0), overlay_color or PRED_COLOR])
        ax.imshow(overlay[crop].astype(np.uint8), cmap=color_map, vmin=0, vmax=1, alpha=0.9,
                  interpolation="nearest")
    elif empty_text:
        ax.text(
            0.5,
            0.08,
            empty_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 3},
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def case_label(selection: str, row: pd.Series, level_name: str) -> str:
    case = textwrap.fill(str(row["case_id"]), width=34, break_long_words=False, break_on_hyphens=False)
    return (
        f"{selection}\n{case}\n{row['group']} | macro Dice={float(row['macro_level_dice']):.3f}"
        f" | shown level={level_name}"
    )


def plot_best_median_worst(
    df: pd.DataFrame,
    images_dir: Path,
    labels_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, list[dict[str, Any]]]:
    valid = df[np.isfinite(pd.to_numeric(df["macro_level_dice"], errors="coerce"))].copy()
    valid["macro_level_dice"] = pd.to_numeric(valid["macro_level_dice"])
    worst = valid.loc[valid["macro_level_dice"].idxmin()]
    best = valid.loc[valid["macro_level_dice"].idxmax()]
    median_value = float(valid["macro_level_dice"].median())
    median = valid.loc[(valid["macro_level_dice"] - median_value).abs().idxmin()]
    selections = [("Best", best), ("Median", median), ("Worst", worst)]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    manifest: list[dict[str, Any]] = []
    for row_index, (selection, row) in enumerate(selections):
        label, level_name = select_representative_level(row)
        image, reference, prediction = load_case(
            str(row["case_id"]), images_dir, labels_dir, predictions_dir
        )
        slice_index = select_slice(reference, prediction, label)
        image_slice = axial_slice(image, slice_index)
        reference_slice = axial_slice(reference == label, slice_index)
        prediction_slice = axial_slice(prediction == label, slice_index)
        crop = square_crop(reference_slice | prediction_slice)
        draw_anatomy_panel(axes[row_index, 0], image_slice, crop)
        draw_anatomy_panel(axes[row_index, 1], image_slice, crop, reference_slice, GT_COLOR)
        draw_anatomy_panel(
            axes[row_index, 2],
            image_slice,
            crop,
            prediction_slice,
            PRED_COLOR,
            empty_text=f"No {level_name} prediction",
        )
        axes[row_index, 0].text(
            -0.08,
            0.5,
            case_label(selection, row, level_name),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
        )
        manifest.append(
            {
                "selection": selection.lower(),
                "case_id": str(row["case_id"]),
                "contrast": str(row["group"]),
                "macro_level_dice": float(row["macro_level_dice"]),
                "shown_level": level_name,
                "canonical_RAS_axial_slice": slice_index,
            }
        )
    for axis, title in zip(axes[0], ["Raw cropped image", "Reference", "Prediction"]):
        axis.set_title(title, fontweight="bold")
    fig.suptitle("Held-out test examples selected by case-level macro Dice", fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor=GT_COLOR, label="Reference level"),
            Patch(facecolor=PRED_COLOR, label="Predicted level"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )
    path = output_dir / "qualitative_best_median_worst.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path, manifest


def plot_t1_reference_cases(
    df: pd.DataFrame,
    images_dir: Path,
    labels_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, list[dict[str, Any]]]:
    affected = df[df["gt_levels"].map(lambda levels: level_is_present(levels, "T1"))].sort_values(
        "case_id"
    )
    if affected.empty:
        raise ValueError("No references contain T1 (label 9).")
    fig, axes = plt.subplots(len(affected), 3, figsize=(14, 3.2 * len(affected)), squeeze=False,
                             constrained_layout=True)
    manifest: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(affected.iterrows()):
        image, reference, prediction = load_case(
            str(row["case_id"]), images_dir, labels_dir, predictions_dir
        )
        slice_index = select_slice(reference, prediction, 9)
        image_slice = axial_slice(image, slice_index)
        reference_slice = axial_slice(reference == 9, slice_index)
        prediction_slice = axial_slice(prediction == 9, slice_index)
        crop = square_crop(reference_slice | prediction_slice)
        draw_anatomy_panel(axes[row_index, 0], image_slice, crop)
        draw_anatomy_panel(axes[row_index, 1], image_slice, crop, reference_slice, GT_COLOR)
        draw_anatomy_panel(
            axes[row_index, 2], image_slice, crop, prediction_slice, PRED_COLOR,
            empty_text="No T1 prediction"
        )
        case = textwrap.fill(str(row["case_id"]), width=34, break_long_words=False,
                             break_on_hyphens=False)
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"{case}\n{row['group']} | macro Dice={float(row['macro_level_dice']):.3f} "
            f"| T1 Dice={float(row['dice_T1']):.3f}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
        )
        manifest.append(
            {
                "case_id": str(row["case_id"]),
                "contrast": str(row["group"]),
                "macro_level_dice": float(row["macro_level_dice"]),
                "t1_dice": float(row["dice_T1"]),
                "canonical_RAS_axial_slice": slice_index,
                "reference_t1_voxels": int(np.sum(reference == 9)),
                "predicted_t1_voxels": int(np.sum(prediction == 9)),
            }
        )
    for axis, title in zip(axes[0], ["Raw cropped image", "T1 reference", "T1 prediction"]):
        axis.set_title(title, fontweight="bold")
    fig.suptitle("T1 audit on all held-out references containing label 9", fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor=GT_COLOR, label="T1 reference"),
            Patch(facecolor=PRED_COLOR, label="T1 prediction"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )
    path = output_dir / "qualitative_t1_reference_cases.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path, manifest


def validate_dataframe(df: pd.DataFrame) -> None:
    required = {
        "case_id",
        "group",
        "gt_levels",
        "pred_levels",
        "macro_level_dice",
        "binary_dice",
        *[f"dice_{name}" for name in LEVELS.values()],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in metrics CSV: {missing}")


def main() -> None:
    args = get_parser().parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(args.metrics_per_case)
    validate_dataframe(dataframe)

    outputs: dict[str, Any] = {
        "metrics_per_case": str(args.metrics_per_case.resolve()),
        "figures": {},
    }
    contrast_path = plot_dice_by_contrast(
        dataframe,
        args.output_dir,
        args.dpi,
        args.model_label,
    )
    level_path = plot_dice_by_level(
        dataframe,
        args.output_dir,
        args.dpi,
        args.model_label,
    )
    outputs["figures"]["dice_by_contrast"] = str(contrast_path.resolve())
    outputs["figures"]["dice_by_spinal_level"] = str(level_path.resolve())

    qualitative_paths = [args.images_dir, args.labels_dir, args.predictions_dir]
    if any(qualitative_paths) and not all(qualitative_paths):
        raise ValueError(
            "Provide --images-dir, --labels-dir, and --predictions-dir together for qualitative figures."
        )
    if all(qualitative_paths):
        best_path, best_manifest = plot_best_median_worst(
            dataframe,
            args.images_dir,
            args.labels_dir,
            args.predictions_dir,
            args.output_dir,
            args.dpi,
        )
        t1_path, t1_manifest = plot_t1_reference_cases(
            dataframe,
            args.images_dir,
            args.labels_dir,
            args.predictions_dir,
            args.output_dir,
            args.dpi,
        )
        outputs["figures"]["qualitative_best_median_worst"] = str(best_path.resolve())
        outputs["figures"]["qualitative_t1_reference_cases"] = str(t1_path.resolve())
        outputs["qualitative_selections"] = best_manifest
        outputs["t1_reference_cases"] = t1_manifest

    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
