#!/usr/bin/env python3
"""Local Streamlit UI for expert dorsal/ventral RootletSeg labeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import to_rgb
from scipy import ndimage

from postprocessing.dorsal_ventral.labeling_app_core import (
    AnnotationCase,
    DiscoveredCase,
    MERGED_DV_CLASS,
    annotate_merged_dv,
    annotate_record,
    discover_reference_label_cases,
    export_annotation_dataset,
    export_nnunet_case,
    load_case,
    merge_saved_annotations,
    read_review_csv,
    split_merged_component_ap,
    write_nnunet_dataset_json,
    write_review_csv,
)


COLORS = {
    "dorsal": "#ff656d",
    "ventral": "#38d2e8",
    "split_dv": "#b39df7",
    "unreviewed": "#6e7b91",
    "selected": "#ffd65c",
}

# This app is for the local RootletSeg annotation workflow.  Point it at the
# checked-out training labels instead of making the reviewer hunt through the
# repository tree on every launch.
_ROOTLETS_WORKSPACE = Path(__file__).resolve().parents[3]
_SOURCE_DATA_ROOT = _ROOTLETS_WORKSPACE / "rootlets-work" / "source-data"
# The multi-subject clone contains git-annex placeholders on this workstation;
# HC-Leipzig is the locally available MRI + reference-label queue.
_DEFAULT_DATASET = _SOURCE_DATA_ROOT / "hc-leipzig-7t-mp2rage"
_DEFAULT_REVIEWER = "kuanw"
_DEFAULT_ANNOTATION_ROOT = (
    _ROOTLETS_WORKSPACE / "results" / "dorsal-ventral" / "annotations-rpi"
)
_DEFAULT_NNUNET_NAME = "Dataset901_RootletDorsalVentral"


def _safe_identifier(value: str, field: str) -> str:
    normalized = value.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not normalized or any(character not in allowed for character in normalized):
        raise ValueError(
            f"{field} may contain only letters, numbers, hyphens, and underscores."
        )
    return normalized


def _review_path(output_directory: Path, case_id: str) -> Path:
    return output_directory / f"{case_id}_desc-rootlet-cluster_review.csv"


def _active_bounds(case: AnnotationCase) -> tuple[slice, slice]:
    active = (case.rootlets_rpi > 0) | case.cord_rpi
    coordinates = np.argwhere(active)
    lower = np.maximum(np.min(coordinates[:, :2], axis=0) - 8, 0)
    upper = np.minimum(
        np.max(coordinates[:, :2], axis=0) + 9,
        np.asarray(case.rootlets_rpi.shape[:2]),
    )
    return slice(int(lower[0]), int(upper[0])), slice(int(lower[1]), int(upper[1]))


def _window(image: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    values = image[support] if np.any(support) else image[np.isfinite(image)]
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _label_by_cluster(records: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(record["cluster_id"]): record["expert_class"] or "unreviewed"
        for record in records
    }


def _cluster_color(cluster_id: int, label: str) -> str:
    """Use one quiet colour for unreviewed support; selection carries focus."""
    del cluster_id
    return COLORS.get(label, COLORS["unreviewed"])


def _best_component_slice(case: AnnotationCase, cluster_id: int) -> int:
    """Pick a slice where the selected 3-D cluster is unquestionably visible."""

    voxels_per_slice = np.count_nonzero(case.cluster_map_rpi == cluster_id, axis=(0, 1))
    if not np.any(voxels_per_slice):
        raise ValueError("The selected rootlet component is absent from the label map.")
    return int(np.argmax(voxels_per_slice))


def _render_slice(
    axis: plt.Axes,
    case: AnnotationCase,
    records: list[dict[str, Any]],
    z_index: int,
    selected_id: int,
    bounds: tuple[slice, slice],
    intensity_window: tuple[float, float],
) -> None:
    x_slice, y_slice = bounds
    anatomy = case.anatomy_rpi[x_slice, y_slice, z_index].T
    clusters = case.cluster_map_rpi[x_slice, y_slice, z_index].T
    cord = case.cord_rpi[x_slice, y_slice, z_index].T
    axis.imshow(
        anatomy,
        cmap="gray",
        origin="upper",
        vmin=intensity_window[0],
        vmax=intensity_window[1],
    )
    labels = _label_by_cluster(records)
    overlay = np.zeros((*clusters.shape, 4), dtype=float)
    for cluster_id in np.unique(clusters[clusters > 0]):
        label = labels[int(cluster_id)]
        selected = int(cluster_id) == selected_id
        overlay[clusters == cluster_id, :3] = to_rgb(
            COLORS["selected"] if selected else _cluster_color(int(cluster_id), label)
        )
        overlay[clusters == cluster_id, 3] = 0.98 if selected else 0.22
    axis.imshow(overlay, origin="upper")
    selected = clusters == selected_id
    if np.any(selected):
        # A contour alone disappears for tiny rootlets.  Dilating it by one
        # pixel gives every selected component an unmistakable white border.
        outline = ndimage.binary_dilation(selected) & ~selected
        border = np.zeros((*selected.shape, 4), dtype=float)
        border[outline, :3] = to_rgb("#ffffff")
        border[outline, 3] = 1.0
        axis.imshow(border, origin="upper")
    if np.any(cord):
        axis.contour(cord, levels=[0.5], colors="#6ee7a8", linewidths=0.7, alpha=0.7)
    axis.text(0.5, 1.01, "A", transform=axis.transAxes, ha="center", color="#c9cedd")
    axis.text(0.5, -0.04, "P", transform=axis.transAxes, ha="center", color="#c9cedd")
    axis.text(-0.03, 0.5, "L", transform=axis.transAxes, va="center", color="#c9cedd")
    axis.text(1.01, 0.5, "R", transform=axis.transAxes, va="center", color="#c9cedd")
    axis.set_xticks([])
    axis.set_yticks([])


def render_focus(
    case: AnnotationCase,
    records: list[dict[str, Any]],
    selected: dict[str, Any],
    z_index: int,
) -> plt.Figure:
    bounds = _active_bounds(case)
    support = np.zeros(case.anatomy_rpi.shape, dtype=bool)
    support[bounds[0], bounds[1], :] = True
    window = _window(case.anatomy_rpi, support)
    figure, axis = plt.subplots(figsize=(6.2, 6.2), facecolor="#11131a")
    axis.set_facecolor("#11131a")
    _render_slice(
        axis,
        case,
        records,
        z_index,
        int(selected["cluster_id"]),
        bounds,
        window,
    )
    figure.tight_layout()
    return figure


def render_montage(
    case: AnnotationCase,
    records: list[dict[str, Any]],
    selected: dict[str, Any],
) -> plt.Figure:
    start = int(selected["slice_start_rpi"])
    stop = int(selected["slice_stop_rpi"])
    slices = np.arange(start, stop + 1, dtype=int)
    bounds = _active_bounds(case)
    support = np.zeros(case.anatomy_rpi.shape, dtype=bool)
    support[bounds[0], bounds[1], :] = True
    window = _window(case.anatomy_rpi, support)
    columns = min(7, len(slices))
    rows = int(np.ceil(len(slices) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.25 * columns, 2.65 * rows),
        facecolor="#11131a",
    )
    for axis, z_index in zip(np.asarray(axes).ravel(), slices):
        _render_slice(
            axis,
            case,
            records,
            int(z_index),
            int(selected["cluster_id"]),
            bounds,
            window,
        )
    for axis in np.asarray(axes).ravel()[len(slices) :]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure


def _next_key(records: list[dict[str, Any]], current_key: str, direction: int) -> str:
    keys = [record["cluster_key"] for record in records]
    current = keys.index(current_key)
    for offset in range(1, len(keys) + 1):
        candidate = records[(current + direction * offset) % len(records)]
        if not candidate["expert_class"]:
            return candidate["cluster_key"]
    return keys[(current + direction) % len(keys)]


def _activate_case(
    source: DiscoveredCase,
    *,
    reviewer_slug: str,
    annotation_root: str,
) -> None:
    """Load one rootlet-labelled volume into the click-to-label workspace."""
    case_slug = _safe_identifier(source.case_id, "Case ID")
    output_directory = Path(annotation_root).expanduser().resolve() / reviewer_slug
    case = load_case(
        source.anatomy_path,
        source.rootlets_path,
        source.cord_path,
        case_id=case_slug,
    )
    review_path = _review_path(output_directory, case.case_id)
    records = merge_saved_annotations(case.records, read_review_csv(review_path))
    if not records:
        raise ValueError("RootletSeg has no labelled rootlet components in this case.")
    st.session_state.case = case
    st.session_state.records = records
    st.session_state.output_directory = output_directory
    st.session_state.reviewer_id = reviewer_slug
    st.session_state.review_path = review_path
    st.session_state.active_source = source
    st.session_state.selected_key = next(
        (
            record["cluster_key"]
            for record in records
            if not record["expert_class"]
        ),
        records[0]["cluster_key"],
    )


def _activate_next_unreviewed_case(
    queue: list[DiscoveredCase], *, start_index: int
) -> bool:
    """Open the next case that still has a rootlet needing a D/V decision."""

    for index in range(start_index, len(queue)):
        _activate_case(
            queue[index],
            reviewer_slug=_DEFAULT_REVIEWER,
            annotation_root=str(_DEFAULT_ANNOTATION_ROOT),
        )
        if any(not record["expert_class"] for record in st.session_state.records):
            st.session_state.queue_index = index
            return True
    return False


def _finish_batch_if_no_unreviewed_case(
    queue: list[DiscoveredCase], *, start_index: int
) -> bool:
    """Open the next unfinished case or export and mark the queue complete."""

    if _activate_next_unreviewed_case(queue, start_index=start_index):
        st.session_state.batch_complete = False
        return True
    _export_labelled_dataset()
    st.session_state.batch_complete = True
    return False


def _default_target_case_count(source_cases: list[DiscoveredCase]) -> int:
    """Resume one case beyond the completed prefix, with a five-case minimum."""

    completed_prefix = 0
    output_directory = _DEFAULT_ANNOTATION_ROOT / _DEFAULT_REVIEWER
    for source in source_cases:
        review_path = _review_path(
            output_directory, _safe_identifier(source.case_id, "Case ID")
        )
        rows = read_review_csv(review_path)
        if not rows or any(not row.get("expert_class") for row in rows):
            break
        completed_prefix += 1
    return min(len(source_cases), max(5, completed_prefix + 1))


def _export_labelled_dataset() -> dict[str, int]:
    """Materialize reviewed clusters and complete cases as training-ready outputs."""
    output_directory: Path = st.session_state.output_directory
    sources: list[DiscoveredCase] = list(
        st.session_state.get("case_queue") or [st.session_state.active_source]
    )
    entries: list[dict[str, Any]] = []
    usable_clusters = 0
    reviewed_clusters = 0
    nnunet_entries: list[dict[str, str]] = []
    nnunet_name = _safe_identifier(
        st.session_state.get(
            "nnunet_dataset_name", "Dataset901_RootletDorsalVentral"
        ),
        "nnU-Net dataset name",
    )
    if not nnunet_name.startswith("Dataset"):
        raise ValueError("nnU-Net dataset name must start with 'Dataset'.")
    nnunet_directory = output_directory / "nnunet_raw" / nnunet_name
    for source in sources:
        case = load_case(
            source.anatomy_path,
            source.rootlets_path,
            source.cord_path,
            case_id=_safe_identifier(source.case_id, "Case ID"),
        )
        review_path = _review_path(output_directory, case.case_id)
        records = merge_saved_annotations(case.records, read_review_csv(review_path))
        trainable = sum(
            2 if record["expert_class"] == MERGED_DV_CLASS else 1
            for record in records
            if record["expert_class"] in {"dorsal", "ventral", MERGED_DV_CLASS}
        )
        reviewed = sum(bool(record["expert_class"]) for record in records)
        if reviewed:
            case_manifest = export_annotation_dataset(
                case,
                output_directory / "dataset" / case.case_id,
                records,
            )
            entries.append(
                {
                    "case_id": case.case_id,
                    "reviewed_clusters": reviewed,
                    "trainable_clusters": trainable,
                    "manifest": str(
                        (
                            output_directory
                            / "dataset"
                            / case.case_id
                            / f"{case.case_id}_annotations.json"
                        ).resolve()
                    ),
                    "outputs": case_manifest["outputs"],
                }
            )
        if records and all(
            record["expert_class"] in {"dorsal", "ventral", MERGED_DV_CLASS}
            for record in records
        ):
            nnunet_entries.append(export_nnunet_case(case, nnunet_directory, records))
        usable_clusters += trainable
        reviewed_clusters += reviewed
    if not usable_clusters:
        raise ValueError("Label at least one rootlet component dorsal or ventral first.")
    dataset_json = (
        write_nnunet_dataset_json(nnunet_directory) if nnunet_entries else None
    )
    manifest_path = output_directory / "dataset" / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "rootlet-dorsal-ventral-dataset-v1",
                "reviewer_id": st.session_state.reviewer_id,
                "source_case_count": len(sources),
                "reviewed_cluster_count": reviewed_clusters,
                "trainable_cluster_count": usable_clusters,
                "cases": entries,
                "nnunet": {
                    "dataset_directory": str(nnunet_directory.resolve()),
                    "dataset_json": str(dataset_json.resolve()) if dataset_json else None,
                    "complete_case_count": len(nnunet_entries),
                    "cases": nnunet_entries,
                    "target_labels": {"background": 0, "dorsal": 1, "ventral": 2},
                    "input_channels": {
                        "0": "MRI",
                        "1": "rootlet label while reviewing; replace with RootletSeg output at inference",
                    },
                },
                "training_contract": (
                    "Cluster records retain partial reviews. nnU-Net cases are written "
                    "only when every rootlet cluster is expert-labelled dorsal or ventral."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "cases": len(entries),
        "reviewed_clusters": reviewed_clusters,
        "trainable_clusters": usable_clusters,
        "nnunet_cases": len(nnunet_entries),
    }


def _amend_last_label() -> bool:
    """Reopen the most recently saved D/V decision, including a prior case."""

    output_directory: Path = st.session_state.output_directory
    queue: list[DiscoveredCase] = st.session_state.get("case_queue", [])
    candidates: list[tuple[str, DiscoveredCase, str]] = []
    for source in queue:
        for row in read_review_csv(_review_path(output_directory, source.case_id)):
            if row.get("expert_class") in {"dorsal", "ventral", MERGED_DV_CLASS}:
                candidates.append((row.get("updated_at", ""), source, row["cluster_key"]))
    if not candidates:
        return False

    _updated_at, source, cluster_key = max(candidates, key=lambda item: item[0])
    _activate_case(
        source,
        reviewer_slug=_DEFAULT_REVIEWER,
        annotation_root=str(_DEFAULT_ANNOTATION_ROOT),
    )
    record = next(
        item for item in st.session_state.records if item["cluster_key"] == cluster_key
    )
    record.update(
        {
            "expert_class": "",
            "split_method": "",
            "split_cut_rpi": "",
            "reviewer_id": "",
            "attachment_visible": "",
            "reviewer_confidence": "",
            "notes": "",
            "updated_at": "",
        }
    )
    write_review_csv(st.session_state.review_path, st.session_state.records)
    st.session_state.queue_index = queue.index(source)
    st.session_state.selected_key = cluster_key
    st.session_state.batch_complete = False
    return True


def _advance_after_save(records: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    """Advance to another unreviewed rootlet or finish the chosen case batch."""

    if all(record["expert_class"] for record in records):
        queue: list[DiscoveredCase] = st.session_state.get("case_queue", [])
        queue_index = int(st.session_state.get("queue_index", 0))
        if queue and _activate_next_unreviewed_case(
            queue, start_index=queue_index + 1
        ):
            return
        _export_labelled_dataset()
        st.session_state.batch_complete = True
        return
    next_key = _next_key(
        records, selected["cluster_key"], 1
    )
    st.session_state.selected_key = next_key
    st.session_state.pending_cluster_key = next_key


def _save_label(expert_class: str) -> None:
    records = st.session_state.records
    selected = next(
        record
        for record in records
        if record["cluster_key"] == st.session_state.selected_key
    )
    annotate_record(
        selected,
        expert_class,
        reviewer_id=st.session_state.reviewer_id,
        visible="yes",
        confidence="medium",
        notes="",
    )
    write_review_csv(st.session_state.review_path, records)
    _advance_after_save(records, selected)


def _save_merged_dv() -> None:
    """Save an expert-declared anterior/posterior split of a merged component."""

    records = st.session_state.records
    selected = next(
        record
        for record in records
        if record["cluster_key"] == st.session_state.selected_key
    )
    _dorsal, _ventral, split_cut = split_merged_component_ap(
        st.session_state.case.cluster_map_rpi, int(selected["cluster_id"])
    )
    annotate_merged_dv(
        selected,
        split_cut_rpi=split_cut,
        reviewer_id=st.session_state.reviewer_id,
    )
    write_review_csv(st.session_state.review_path, records)
    _advance_after_save(records, selected)


st.set_page_config(
    page_title="Rootlet Atlas · Dorsal/Ventral Review",
    page_icon="🧠",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #0c0e14; color: #eef1f7; }
    [data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
    [data-testid="stBaseButton-elementToolbar"] { display: none; }
    .block-container { padding-top: 0.75rem; max-width: 1500px; }
    .status-card { background: #151923; border: 1px solid #272c3a; border-radius: 14px;
      padding: 0.85rem 1rem; margin-bottom: 0.8rem; }
    .small-copy { color: #9ca6ba; font-size: 0.88rem; }
    .queue-status { color: #aab4c8; font-size: 0.9rem; margin: 0 0 0.5rem 0.15rem; }
    div.stButton > button { min-height: 3rem; border-radius: 10px; font-weight: 700; }
    [data-testid="stBaseButton-secondary"] { background: #38d2e8; border-color: #38d2e8; color: #071317; }
    </style>
    """,
    unsafe_allow_html=True,
)

# A live Streamlit reload can retain a pre-RPI case object from the preceding
# version.  Discard only that in-memory view; the CSV annotations remain intact.
if "case" in st.session_state and not hasattr(
    st.session_state.case, "cluster_map_rpi"
):
    for session_key in (
        "case",
        "records",
        "active_source",
        "selected_key",
        "pending_cluster_key",
        "queue_index",
    ):
        st.session_state.pop(session_key, None)

if "source_cases" not in st.session_state:
    try:
        source_cases = discover_reference_label_cases(_DEFAULT_DATASET)
        if not source_cases:
            raise ValueError("The local MRI/rootlet-label pairs were not found.")
        st.session_state.source_cases = source_cases
        st.session_state.annotation_root = str(_DEFAULT_ANNOTATION_ROOT)
        st.session_state.nnunet_dataset_name = _DEFAULT_NNUNET_NAME
        st.session_state.target_case_count = _default_target_case_count(source_cases)
    except Exception as error:
        st.error(f"Could not open the local rootlet-label queue: {error}")
        st.stop()

if "case" not in st.session_state:
    try:
        st.session_state.case_queue = st.session_state.source_cases[
            : st.session_state.target_case_count
        ]
        st.session_state.batch_complete = False
        _finish_batch_if_no_unreviewed_case(
            st.session_state.case_queue,
            start_index=0,
        )
    except Exception as error:
        st.error(f"Could not open the local rootlet-label queue: {error}")
        st.stop()

case: AnnotationCase = st.session_state.case
records: list[dict[str, Any]] = st.session_state.records
queue = st.session_state.get("case_queue", [])
queue_index = int(st.session_state.get("queue_index", 0))
source_cases: list[DiscoveredCase] = st.session_state.source_cases
case_count_column, status_column = st.columns([1, 6])
with case_count_column:
    target_case_count = st.number_input(
        "Cases",
        min_value=1,
        max_value=len(source_cases),
        step=1,
        key="target_case_count",
    )
desired_queue = source_cases[: int(target_case_count)]
if desired_queue != queue:
    st.session_state.case_queue = desired_queue
    st.session_state.batch_complete = False
    _finish_batch_if_no_unreviewed_case(desired_queue, start_index=0)
    st.rerun()
if st.session_state.get("batch_complete"):
    st.success("All queued rootlet clusters are labelled and exported.")
    if st.button("← Amend last", width="content"):
        if _amend_last_label():
            st.rerun()
    st.stop()
ordered_records = sorted(
    records,
    key=lambda record: (
        bool(record["expert_class"]),
        int(record["level"]),
        record["side"],
        int(record["component_id"]),
    ),
)
ordered_records = [record for record in ordered_records if not record["expert_class"]]
if not ordered_records:
    st.info("This case is complete; loading the next case.")
    st.stop()
available_keys = [record["cluster_key"] for record in ordered_records]
pending_key = st.session_state.pop("pending_cluster_key", None)
if pending_key in available_keys:
    st.session_state.selected_key = pending_key
if st.session_state.selected_key not in available_keys:
    st.session_state.selected_key = available_keys[0]
selected_key = st.session_state.selected_key
selected = next(record for record in records if record["cluster_key"] == selected_key)
case_position = f"{queue_index + 1}/{len(queue)}" if queue else "1/1"
rootlet_position = f"{len(records) - len(ordered_records) + 1}/{len(records)}"
with status_column:
    st.markdown(
        f'<div class="queue-status">case {case_position} &nbsp;·&nbsp; rootlet {rootlet_position}</div>',
        unsafe_allow_html=True,
    )

viewer, inspector = st.columns([4.5, 1], gap="large")
with viewer:
    z_index = _best_component_slice(case, int(selected["cluster_id"]))
    focus_figure = render_focus(case, records, selected, z_index)
    st.pyplot(focus_figure, width="stretch")
    plt.close(focus_figure)
    montage = render_montage(case, records, selected)
    st.pyplot(montage, width="stretch")
    plt.close(montage)

with inspector:
    st.markdown(
        """
        <div class="status-card">
          <b>Yellow = this rootlet.</b><br/>
          <span class="small-copy">If it contains a top ventral and bottom dorsal pair,
          use Split. Otherwise click its label.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Amend last", width="stretch"):
        if _amend_last_label():
            st.rerun()
    if st.button("SPLIT: TOP VENTRAL / BOTTOM DORSAL", width="stretch"):
        try:
            _save_merged_dv()
            st.rerun()
        except ValueError as error:
            st.error(str(error))
    if st.button("DORSAL", width="stretch", type="primary"):
        _save_label("dorsal")
        st.rerun()
    if st.button("VENTRAL", width="stretch", type="secondary"):
        _save_label("ventral")
        st.rerun()
