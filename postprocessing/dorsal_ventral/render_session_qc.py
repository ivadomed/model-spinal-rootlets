#!/usr/bin/env python3
"""Render paired axial overlays for qualitative session-robustness review."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def _resolve(value: str, directory: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else directory / path


def read_qc_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject", "session", "image", "combined", "dorsal", "ventral"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing QC columns: {', '.join(sorted(missing))}")
        rows = [dict(row) for row in reader]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for field in ("image", "combined", "dorsal", "ventral"):
            row[field] = str(_resolve(row[field], path.parent).resolve())
        grouped[row["subject"]].append(row)
    return grouped


def _canonical_data(path: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(path))
    return image, np.asanyarray(image.dataobj)


def _load_scan(row: dict[str, str]) -> dict[str, Any]:
    image, anatomical = _canonical_data(row["image"])
    arrays: dict[str, np.ndarray] = {"image": anatomical}
    for field in ("combined", "dorsal", "ventral"):
        candidate, data = _canonical_data(row[field])
        if candidate.shape != image.shape or not np.allclose(
            candidate.affine, image.affine, atol=1e-4
        ):
            raise ValueError(f"{field} grid differs from image for {row['subject']} {row['session']}")
        arrays[field] = np.rint(data).astype(np.int16)
    if np.any((arrays["dorsal"] > 0) & (arrays["ventral"] > 0)) or not np.array_equal(
        arrays["dorsal"] + arrays["ventral"], arrays["combined"]
    ):
        raise ValueError(f"Invalid D/V partition for {row['subject']} {row['session']}")
    arrays.update({"subject": row["subject"], "session": row["session"]})
    return arrays


def _slice_indices(combined: np.ndarray, panels: int) -> list[int]:
    support = np.flatnonzero(np.any(combined > 0, axis=(0, 1)))
    if not support.size:
        raise ValueError("Cannot render an empty combined rootlet mask.")
    positions = np.linspace(0, support.size - 1, panels)
    return [int(support[int(round(position))]) for position in positions]


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, (1, 99))
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def render_subject(rows: list[dict[str, str]], output: Path, panels: int = 6) -> None:
    scans = [_load_scan(row) for row in sorted(rows, key=lambda item: item["session"])]
    if len(scans) != 2:
        raise ValueError(f"Expected two sessions for {rows[0]['subject']}; found {len(scans)}.")
    figure, axes = plt.subplots(
        len(scans), panels, figsize=(2.7 * panels, 2.9 * len(scans)), squeeze=False
    )
    for row_index, scan in enumerate(scans):
        lower, upper = _display_limits(scan["image"])
        for column, z_index in enumerate(_slice_indices(scan["combined"], panels)):
            axis = axes[row_index, column]
            background = scan["image"][:, :, z_index].T
            dorsal = scan["dorsal"][:, :, z_index].T > 0
            ventral = scan["ventral"][:, :, z_index].T > 0
            levels = sorted(
                int(value)
                for value in np.unique(scan["combined"][:, :, z_index])
                if value > 0
            )
            axis.imshow(background, cmap="gray", origin="lower", vmin=lower, vmax=upper)
            axis.imshow(
                np.ma.masked_where(~dorsal, dorsal),
                cmap=matplotlib.colors.ListedColormap([(1.0, 0.1, 0.1, 0.65)]),
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axis.imshow(
                np.ma.masked_where(~ventral, ventral),
                cmap=matplotlib.colors.ListedColormap([(0.0, 0.85, 1.0, 0.65)]),
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axis.set_title(
                f"{scan['session']} · z={z_index} · L{','.join(map(str, levels))}", fontsize=8
            )
            axis.axis("off")
    figure.suptitle(
        f"{scans[0]['subject']} paired-session QC (not registered; not accuracy)", fontsize=12
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
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(manifest: Path, output_directory: Path, panels: int = 6) -> list[Path]:
    if panels < 2:
        raise ValueError("panels must be at least 2.")
    grouped = read_qc_manifest(manifest)
    outputs: list[Path] = []
    for subject, rows in sorted(grouped.items()):
        output = output_directory / f"{subject}_paired-session_qc.png"
        render_subject(rows, output, panels=panels)
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
