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

from postprocessing.dorsal_ventral.labeling_app_core import (
    EXPERT_CLASSES,
    AnnotationCase,
    DiscoveredCase,
    annotate_record,
    discover_cases,
    export_annotation_dataset,
    load_case,
    merge_saved_annotations,
    read_review_csv,
    write_review_csv,
)


COLORS = {
    "dorsal": "#ff656d",
    "ventral": "#38d2e8",
    "mixed": "#ffbf47",
    "unclear": "#ad8cff",
    "unreviewed": "#f6f7fb",
}
COMPONENT_COLORS = (
    "#ffbe5c",
    "#8bd3ff",
    "#c9a7ff",
    "#79e2c0",
    "#ff9ab1",
    "#c6e36b",
    "#f0a4ff",
    "#73c7ff",
    "#f4d36b",
    "#9bb7ff",
)


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
    active = (case.rootlets_ras > 0) | case.cord_ras
    coordinates = np.argwhere(active)
    lower = np.maximum(np.min(coordinates[:, :2], axis=0) - 8, 0)
    upper = np.minimum(
        np.max(coordinates[:, :2], axis=0) + 9,
        np.asarray(case.rootlets_ras.shape[:2]),
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
    """Keep unreviewed 3-D components visually distinct in the viewer."""
    if label != "unreviewed":
        return COLORS[label]
    return COMPONENT_COLORS[(cluster_id - 1) % len(COMPONENT_COLORS)]


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
    anatomy = case.anatomy_ras[x_slice, y_slice, z_index].T
    clusters = case.cluster_map_ras[x_slice, y_slice, z_index].T
    cord = case.cord_ras[x_slice, y_slice, z_index].T
    axis.imshow(
        anatomy,
        cmap="gray",
        origin="lower",
        vmin=intensity_window[0],
        vmax=intensity_window[1],
    )
    labels = _label_by_cluster(records)
    overlay = np.zeros((*clusters.shape, 4), dtype=float)
    for cluster_id in np.unique(clusters[clusters > 0]):
        label = labels[int(cluster_id)]
        selected = int(cluster_id) == selected_id
        overlay[clusters == cluster_id, :3] = to_rgb(
            _cluster_color(int(cluster_id), label)
        )
        overlay[clusters == cluster_id, 3] = 0.92 if selected else 0.34
    axis.imshow(overlay, origin="lower")
    selected = clusters == selected_id
    if np.any(selected):
        axis.contour(selected, levels=[0.5], colors="#ffffff", linewidths=1.2)
    if np.any(cord):
        axis.contour(cord, levels=[0.5], colors="#6ee7a8", linewidths=0.7, alpha=0.7)
    axis.text(0.5, 1.01, "A", transform=axis.transAxes, ha="center", color="#c9cedd")
    axis.text(0.5, -0.04, "P", transform=axis.transAxes, ha="center", color="#c9cedd")
    axis.text(-0.03, 0.5, "L", transform=axis.transAxes, va="center", color="#c9cedd")
    axis.text(1.01, 0.5, "R", transform=axis.transAxes, va="center", color="#c9cedd")
    axis.set_title(f"RAS axial · slice {z_index}", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def render_focus(
    case: AnnotationCase,
    records: list[dict[str, Any]],
    selected: dict[str, Any],
    z_index: int,
) -> plt.Figure:
    bounds = _active_bounds(case)
    support = np.zeros(case.anatomy_ras.shape, dtype=bool)
    support[bounds[0], bounds[1], :] = True
    window = _window(case.anatomy_ras, support)
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
    start = int(selected["slice_start_ras"])
    stop = int(selected["slice_stop_ras"])
    slices = np.arange(start, stop + 1, dtype=int)
    bounds = _active_bounds(case)
    support = np.zeros(case.anatomy_ras.shape, dtype=bool)
    support[bounds[0], bounds[1], :] = True
    window = _window(case.anatomy_ras, support)
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
    """Load one model-segmented volume into the click-to-label workspace."""
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
    st.session_state.cluster_selector = st.session_state.selected_key


def _export_labelled_dataset() -> dict[str, int]:
    """Materialize reviewed D/V components as a multi-case training dataset."""
    output_directory: Path = st.session_state.output_directory
    sources: list[DiscoveredCase] = list(
        st.session_state.get("case_queue") or [st.session_state.active_source]
    )
    entries: list[dict[str, Any]] = []
    usable_clusters = 0
    reviewed_clusters = 0
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
            record["expert_class"] in {"dorsal", "ventral"}
            for record in records
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
        usable_clusters += trainable
        reviewed_clusters += reviewed
    if not usable_clusters:
        raise ValueError("Label at least one rootlet component dorsal or ventral first.")
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
                "training_contract": (
                    "Only expert dorsal and ventral components are training "
                    "targets; mixed, unclear, and unreviewed components are excluded."
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
    }


def _save_label(expert_class: str) -> None:
    records = st.session_state.records
    selected = next(
        record
        for record in records
        if record["cluster_key"] == st.session_state.selected_key
    )
    visible = st.session_state.get("visibility", "yes")
    confidence = st.session_state.get("confidence", "medium")
    notes = st.session_state.get("notes", "")
    annotate_record(
        selected,
        expert_class,
        reviewer_id=st.session_state.reviewer_id,
        visible=visible,
        confidence=confidence,
        notes=notes,
    )
    write_review_csv(st.session_state.review_path, records)
    if all(record["expert_class"] for record in records):
        queue: list[DiscoveredCase] = st.session_state.get("case_queue", [])
        queue_index = int(st.session_state.get("queue_index", 0))
        if queue and queue_index + 1 < len(queue):
            next_index = queue_index + 1
            _activate_case(
                queue[next_index],
                reviewer_slug=st.session_state.reviewer_id,
                annotation_root=st.session_state.annotation_root,
            )
            st.session_state.queue_index = next_index
            st.session_state.case_advance_notice = (
                f"Case complete. Now labeling {queue[next_index].case_id}."
            )
            return
        st.session_state.batch_complete = True
    next_key = _next_key(
        records, selected["cluster_key"], 1
    )
    st.session_state.selected_key = next_key
    st.session_state.pending_cluster_key = next_key


st.set_page_config(
    page_title="Rootlet Atlas · Dorsal/Ventral Review",
    page_icon="🧠",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #0c0e14; color: #eef1f7; }
    [data-testid="stSidebar"] { background: #12151d; }
    .block-container { padding-top: 1.4rem; max-width: 1500px; }
    .status-card { background: #151923; border: 1px solid #272c3a; border-radius: 14px;
      padding: 0.85rem 1rem; margin-bottom: 0.8rem; }
    .small-copy { color: #9ca6ba; font-size: 0.88rem; }
    div.stButton > button { min-height: 3rem; border-radius: 10px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Rootlet Atlas")
st.caption(
    "Expert review of 3-D RootletSeg clusters · anterior/ventral is shown at the top in RAS axial view"
)

with st.sidebar:
    st.subheader("Build a dorsal / ventral label set")
    reviewer_id = st.text_input("Reviewer ID", value="reviewer-01")
    output_value = st.text_input(
        "Annotation directory",
        value=str(Path("results/dorsal-ventral/annotations").resolve()),
    )
    st.divider()
    default_search_root = Path.cwd().parent / "results" / "dorsal-ventral"
    search_value = st.text_input(
        "Dataset folder",
        value=str(default_search_root.resolve()),
        help="Searches locally for the first model's RootletSeg labels and matching images.",
    )
    if st.button("Find RootletSeg cases", width="stretch"):
        try:
            st.session_state.discovered_cases = discover_cases(Path(search_value))
            if not st.session_state.discovered_cases:
                st.warning("No complete anatomy, RootletSeg, and cord triplets found.")
        except Exception as error:
            st.error(str(error))

    discovered_cases = st.session_state.get("discovered_cases", [])
    if discovered_cases:
        selected_case = st.selectbox(
            f"{len(discovered_cases)} cases ready",
            discovered_cases,
            format_func=lambda case: case.case_id,
        )
        queue_column, one_case_column = st.columns(2)
        start_queue = queue_column.button(
            f"Label all {len(discovered_cases)}", width="stretch", type="primary"
        )
        start_one = one_case_column.button("One case", width="stretch")
        if start_queue or start_one:
            try:
                reviewer_slug = _safe_identifier(reviewer_id, "Reviewer ID")
                queue = list(discovered_cases) if start_queue else [selected_case]
                st.session_state.case_queue = queue
                st.session_state.queue_index = 0
                st.session_state.annotation_root = output_value
                st.session_state.batch_complete = False
                with st.spinner("Building 3-D rootlet components…"):
                    _activate_case(
                        queue[0],
                        reviewer_slug=reviewer_slug,
                        annotation_root=output_value,
                    )
                st.success(
                    f"Ready: {len(st.session_state.records)} components in "
                    f"{queue[0].case_id}."
                )
            except Exception as error:
                st.error(str(error))

    with st.expander("Advanced: load one case by file path"):
        if "case_id_input" not in st.session_state:
            st.session_state.case_id_input = "sub-001"
        case_id = st.text_input("Case ID", key="case_id_input")
        anatomy_value = st.text_input("Anatomical NIfTI path", key="anatomy_path_input")
        rootlets_value = st.text_input("RootletSeg NIfTI path", key="rootlets_path_input")
        cord_value = st.text_input("Spinal cord mask path", key="cord_path_input")
        if st.button("Load one case", width="stretch"):
            try:
                reviewer_slug = _safe_identifier(reviewer_id, "Reviewer ID")
                source = DiscoveredCase(
                    case_id=_safe_identifier(case_id, "Case ID"),
                    anatomy_path=Path(anatomy_value),
                    rootlets_path=Path(rootlets_value),
                    cord_path=Path(cord_value),
                )
                st.session_state.case_queue = []
                st.session_state.queue_index = 0
                st.session_state.annotation_root = output_value
                st.session_state.batch_complete = False
                with st.spinner("Building 3-D rootlet components…"):
                    _activate_case(
                        source,
                        reviewer_slug=reviewer_slug,
                        annotation_root=output_value,
                    )
                st.success(f"Loaded {len(st.session_state.records)} components.")
            except Exception as error:
                st.error(str(error))

    show_suggestion = st.checkbox(
        "Show model suggestion",
        value=False,
        help="Keep this off for blinded ground-truth review.",
    )
if "case" not in st.session_state:
    st.markdown(
        """
        <div class="status-card">
          <b>Make the first dorsal/ventral dataset.</b><br/>
          <span class="small-copy">Find the first model's RootletSeg cases, start the queue, then assign every
          coloured 3-D component dorsal or ventral. Decisions are saved locally after every click.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

case: AnnotationCase = st.session_state.case
records: list[dict[str, Any]] = st.session_state.records
if notice := st.session_state.pop("case_advance_notice", None):
    st.success(notice)
reviewed = sum(bool(record["expert_class"]) for record in records)
dorsal_count = sum(record["expert_class"] == "dorsal" for record in records)
ventral_count = sum(record["expert_class"] == "ventral" for record in records)

metric_columns = st.columns(4)
queue = st.session_state.get("case_queue", [])
queue_index = int(st.session_state.get("queue_index", 0))
case_position = f"{queue_index + 1}/{len(queue)}" if queue else "single case"
metric_columns[0].metric("This case", f"{reviewed}/{len(records)}")
metric_columns[1].metric("Dorsal", dorsal_count)
metric_columns[2].metric("Ventral", ventral_count)
metric_columns[3].metric("Queue", case_position)
st.progress(reviewed / len(records))

filter_column, navigation_column, export_column = st.columns([2, 1, 1])
with filter_column:
    display_filter = st.selectbox(
        "Review queue",
        ("Unreviewed first", "All clusters", "Dorsal", "Ventral", "Mixed / unclear"),
        label_visibility="collapsed",
    )
with navigation_column:
    if st.button("Previous", width="stretch"):
        previous_key = _next_key(
            records, st.session_state.selected_key, -1
        )
        st.session_state.selected_key = previous_key
        st.session_state.cluster_selector = previous_key
        st.rerun()
with export_column:
    if st.button("Confirm dataset", width="stretch", type="primary"):
        try:
            write_review_csv(st.session_state.review_path, records)
            with st.spinner("Writing the reviewed multi-case dataset…"):
                dataset = _export_labelled_dataset()
            st.success(
                f"Dataset confirmed: {dataset['trainable_clusters']} dorsal/ventral "
                f"components from {dataset['cases']} cases."
            )
        except Exception as error:
            st.error(str(error))

ordered_records = sorted(
    records,
    key=lambda record: (
        bool(record["expert_class"]),
        int(record["level"]),
        record["side"],
        int(record["component_id"]),
    ),
) if display_filter == "Unreviewed first" else list(records)
if display_filter == "Dorsal":
    ordered_records = [r for r in ordered_records if r["expert_class"] == "dorsal"]
elif display_filter == "Ventral":
    ordered_records = [r for r in ordered_records if r["expert_class"] == "ventral"]
elif display_filter == "Mixed / unclear":
    ordered_records = [
        r for r in ordered_records if r["expert_class"] in {"mixed", "unclear"}
    ]
if not ordered_records:
    st.info("This queue is empty.")
    st.stop()
available_keys = [record["cluster_key"] for record in ordered_records]
pending_key = st.session_state.pop("pending_cluster_key", None)
if pending_key in available_keys:
    st.session_state.selected_key = pending_key
    st.session_state.cluster_selector = pending_key
if st.session_state.selected_key not in available_keys:
    st.session_state.selected_key = available_keys[0]
if st.session_state.get("cluster_selector") not in available_keys:
    st.session_state.cluster_selector = st.session_state.selected_key
selected_key = st.selectbox(
    "Select coloured component",
    available_keys,
    key="cluster_selector",
    format_func=lambda key: next(
        f"{key} · {record['voxel_count']} voxels · "
        f"{record['expert_class'] or 'unreviewed'}"
        for record in records
        if record["cluster_key"] == key
    ),
)
st.session_state.selected_key = selected_key
selected = next(record for record in records if record["cluster_key"] == selected_key)
if st.session_state.get("editor_cluster") != selected_key:
    st.session_state.editor_cluster = selected_key
    st.session_state.visibility = selected["attachment_visible"] or "yes"
    st.session_state.confidence = selected["reviewer_confidence"] or "medium"
    st.session_state.notes = selected["notes"]

viewer, inspector = st.columns([1.65, 1], gap="large")
with viewer:
    slice_min = 0
    slice_max = case.rootlets_ras.shape[2] - 1
    center = int(round(float(selected["centroid_z_ras"])))
    z_index = st.slider(
        "Whole-case axial slice",
        slice_min,
        slice_max,
        min(max(center, slice_min), slice_max),
        key=f"slice-{selected_key}",
    )
    st.caption(
        "Every unreviewed 3-D RootletSeg component has a distinct colour. "
        "After assignment, dorsal is coral and ventral is cyan."
    )
    focus_figure = render_focus(case, records, selected, z_index)
    st.pyplot(focus_figure, width="stretch")
    plt.close(focus_figure)
    with st.expander("Show this component across all of its slices", expanded=True):
        montage = render_montage(case, records, selected)
        st.pyplot(montage, width="stretch")
        plt.close(montage)

with inspector:
    current = selected["expert_class"] or "unreviewed"
    suggestion_text = selected["suggested_class"] if show_suggestion else "hidden"
    st.markdown(
        f"""
        <div class="status-card">
          <b>{selected['cluster_key']}</b> · current: <span style="color:{COLORS[current]}">{current}</span><br/>
          <span class="small-copy">Level {selected['level']} · {selected['side']} ·
          slices {selected['slice_start_ras']}–{selected['slice_stop_ras']} ·
          suggested {suggestion_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    first_row = st.columns(2)
    if first_row[0].button(
        "Dorsal", width="stretch", type="primary"
    ):
        _save_label("dorsal")
        st.rerun()
    if first_row[1].button("Ventral", width="stretch", type="primary"):
        _save_label("ventral")
        st.rerun()
    with st.expander("Flag a difficult component"):
        st.selectbox(
            "Attachment visible?",
            ("yes", "no", "unclear"),
            key="visibility",
        )
        st.selectbox(
            "Confidence",
            ("high", "medium", "low"),
            key="confidence",
        )
        st.text_area("Notes", key="notes", height=90)
        second_row = st.columns(2)
        if second_row[0].button("Mixed", width="stretch"):
            _save_label("mixed")
            st.rerun()
        if second_row[1].button("Unclear", width="stretch"):
            _save_label("unclear")
            st.rerun()
    st.caption(
        "Click dorsal or ventral to save and advance. Mixed and unclear are "
        "optional QC flags and are excluded from training."
    )

with st.expander("Review table"):
    table_rows = []
    for record in records:
        row = {
            "cluster": record["cluster_key"],
            "level": record["level"],
            "side": record["side"],
            "voxels": record["voxel_count"],
            "expert": record["expert_class"] or "—",
            "visible": record["attachment_visible"] or "—",
            "confidence": record["reviewer_confidence"] or "—",
        }
        if show_suggestion:
            row["suggestion"] = record["suggested_class"]
        table_rows.append(row)
    st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
    )
