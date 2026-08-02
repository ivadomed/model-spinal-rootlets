#!/usr/bin/env python3
"""Build and optimize a GNN-ready graph of proximal attachment islands.

The graph never uses deterministic ``predicted_class`` values as node features
or targets. Expert labels are optional and stored separately as -1/0/1 so the
same serialization can support label-free optimization now and supervised,
subject-grouped evaluation later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CLASS_TO_INDEX = {"dorsal": 0, "ventral": 1}
EDGE_TYPES = {"competition": 0, "bilateral": 1, "adjacent_level": 2}
REQUIRED_FIELDS = {
    "subject",
    "attachment_id",
    "level",
    "side",
    "component_id",
    "median_ap_mm",
    "minimum_cord_distance_mm",
    "eligible_for_splitter",
    "voxel_count",
    "predicted_class",
    "expert_class",
}


def read_attachment_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        missing = REQUIRED_FIELDS - set(fields)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no attachment rows.")
    return fields, rows


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true/false, got {value!r}.")
    return normalized == "true"


def _node(row: dict[str, str], index: int) -> dict[str, Any]:
    side = row["side"].lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Invalid side: {row['side']!r}.")
    expert_class = row["expert_class"].lower()
    if expert_class and expert_class not in {*CLASS_TO_INDEX, "unclear"}:
        raise ValueError(f"Invalid expert_class: {expert_class!r}.")
    level = int(row["level"])
    ap = float(row["median_ap_mm"])
    cord_distance = float(row["minimum_cord_distance_mm"])
    voxel_count = int(row["voxel_count"])
    if not all(math.isfinite(value) for value in (ap, cord_distance)):
        raise ValueError("Graph features must be finite.")
    if voxel_count <= 0:
        raise ValueError("voxel_count must be positive.")
    return {
        "index": index,
        "attachment_id": row["attachment_id"],
        "subject": row["subject"],
        "level": level,
        "side": side,
        "component_id": int(row["component_id"]),
        "features": {
            "level": level,
            "side_right": int(side == "right"),
            "median_ap_mm": ap,
            "minimum_cord_distance_mm": cord_distance,
            "log1p_voxel_count": math.log1p(voxel_count),
            "eligible_for_splitter": int(
                _parse_bool(row["eligible_for_splitter"])
            ),
        },
        "expert_label": CLASS_TO_INDEX.get(expert_class, -1),
    }


def _ranked(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        nodes,
        key=lambda node: (
            node["features"]["median_ap_mm"],
            node["attachment_id"],
        ),
    )


def _rank_edges(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    edge_type: str,
) -> list[dict[str, Any]]:
    """Connect ordinally similar attachments across sides or adjacent levels."""
    first = _ranked(first)
    second = _ranked(second)
    edges: list[dict[str, Any]] = []
    for i, left in enumerate(first):
        left_rank = (i + 0.5) / len(first)
        for j, right in enumerate(second):
            right_rank = (j + 0.5) / len(second)
            rank_distance = abs(left_rank - right_rank)
            weight = math.exp(-3.0 * rank_distance)
            edges.append(
                {
                    "source": left["index"],
                    "target": right["index"],
                    "type": edge_type,
                    "type_index": EDGE_TYPES[edge_type],
                    "weight": weight,
                }
            )
    return edges


def build_attachment_graph(
    rows: Iterable[dict[str, str]], *, group_id: str | None = None
) -> dict[str, Any]:
    rows = list(rows)
    nodes = [_node(row, index) for index, row in enumerate(rows)]
    attachment_ids = [node["attachment_id"] for node in nodes]
    if len(set(attachment_ids)) != len(attachment_ids):
        raise ValueError("attachment_id values must be unique within one graph.")

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[(node["level"], node["side"])].append(node)

    edges: list[dict[str, Any]] = []
    for group_nodes in grouped.values():
        ordered = _ranked(group_nodes)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                edges.append(
                    {
                        "source": first["index"],
                        "target": second["index"],
                        "type": "competition",
                        "type_index": EDGE_TYPES["competition"],
                        "weight": 1.0,
                    }
                )

    levels = sorted({node["level"] for node in nodes})
    for level in levels:
        right = grouped.get((level, "right"), [])
        left = grouped.get((level, "left"), [])
        if right and left:
            edges.extend(_rank_edges(right, left, "bilateral"))
    for side in ("right", "left"):
        for level in levels:
            current = grouped.get((level, side), [])
            following = grouped.get((level + 1, side), [])
            if current and following:
                edges.extend(_rank_edges(current, following, "adjacent_level"))

    subjects = sorted({node["subject"] for node in nodes})
    if len(subjects) != 1:
        raise ValueError("Each graph must contain exactly one scan subject identifier.")
    return {
        "schema": "rootlet-attachment-graph-v1",
        "subject": subjects[0],
        "group_id": group_id or subjects[0],
        "feature_names": [
            "level",
            "side_right",
            "median_ap_mm",
            "minimum_cord_distance_mm",
            "log1p_voxel_count",
            "eligible_for_splitter",
        ],
        "edge_types": EDGE_TYPES,
        "edge_semantics": "undirected; expand each stored edge in both directions",
        "nodes": nodes,
        "edges": edges,
        "provenance": {
            "uses_input_predicted_class": False,
            "expert_labels_are_optional_targets_only": True,
        },
    }


def _unary_costs(values: list[float]) -> dict[int, float]:
    """Return costs indexed by the count assigned to posterior/dorsal."""
    values = sorted(values)
    count = len(values)
    if count == 1:
        ap = values[0]
        scale = 0.5
        return {0: max(-ap, 0.0) / scale, 1: max(ap, 0.0) / scale}
    gaps = [
        values[index + 1] - values[index]
        for index in range(count - 1)
    ]
    maximum_gap = max(max(gaps), 1e-6)
    return {
        split: (maximum_gap - gaps[split - 1]) / maximum_gap
        + 0.2 * abs(split / count - 0.5)
        for split in range(1, count)
    }


def optimize_ordered_groups(
    grouped_values: dict[tuple[int, str], list[float]],
    *,
    bilateral_weight: float = 0.8,
    level_weight: float = 0.6,
) -> dict[tuple[int, str], int]:
    """Return the posterior/dorsal count for every ordered side/level group."""
    grouped_values = {
        key: list(values) for key, values in grouped_values.items() if values
    }
    costs = {key: _unary_costs(values) for key, values in grouped_values.items()}
    state = {
        key: min(options, key=lambda value: (options[value], value))
        for key, options in costs.items()
    }

    def fraction(key: tuple[int, str], candidate: int | None = None) -> float:
        return (state[key] if candidate is None else candidate) / len(
            grouped_values[key]
        )

    keys = sorted(grouped_values)
    for _ in range(50):
        changed = False
        for key in keys:
            level, side = key
            opposite = (level, "left" if side == "right" else "right")
            neighbours = [
                adjacent
                for adjacent in ((level - 1, side), (level + 1, side))
                if adjacent in grouped_values
            ]

            def energy(candidate: int) -> float:
                value = costs[key][candidate]
                if opposite in grouped_values:
                    value += bilateral_weight * abs(
                        fraction(key, candidate) - fraction(opposite)
                    )
                value += level_weight * sum(
                    abs(fraction(key, candidate) - fraction(neighbour))
                    for neighbour in neighbours
                )
                return value

            best = min(costs[key], key=lambda value: (energy(value), value))
            if best != state[key]:
                state[key] = best
                changed = True
        if not changed:
            break
    return state


def optimize_attachment_graph(
    graph: dict[str, Any], *, bilateral_weight: float = 0.8, level_weight: float = 0.6
) -> dict[str, str]:
    """Minimize a small graph energy over ordered side/level split states."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for node in graph["nodes"]:
        grouped[(node["level"], node["side"])].append(node)
    state = optimize_ordered_groups(
        {
            key: [node["features"]["median_ap_mm"] for node in nodes]
            for key, nodes in grouped.items()
        },
        bilateral_weight=bilateral_weight,
        level_weight=level_weight,
    )

    predictions: dict[str, str] = {}
    for key, nodes in grouped.items():
        ordered = _ranked(nodes)
        dorsal_count = state[key]
        for index, node in enumerate(ordered):
            predictions[node["attachment_id"]] = (
                "dorsal" if index < dorsal_count else "ventral"
            )
    return predictions


def write_predicted_review(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    predictions: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["predicted_class"] = predictions[row["attachment_id"]]
            writer.writerow(output)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-graph", required=True)
    parser.add_argument("--output-review", required=True)
    parser.add_argument("--group-id")
    parser.add_argument("--bilateral-weight", type=float, default=0.8)
    parser.add_argument("--level-weight", type=float, default=0.6)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    fields, rows = read_attachment_rows(Path(args.review_csv))
    graph = build_attachment_graph(rows, group_id=args.group_id)
    predictions = optimize_attachment_graph(
        graph,
        bilateral_weight=args.bilateral_weight,
        level_weight=args.level_weight,
    )
    output_graph = Path(args.output_graph)
    output_graph.parent.mkdir(parents=True, exist_ok=True)
    output_graph.write_text(json.dumps(graph, indent=2) + "\n")
    write_predicted_review(
        Path(args.output_review), fields, rows, predictions
    )
    print(
        json.dumps(
            {
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "dorsal": sum(value == "dorsal" for value in predictions.values()),
                "ventral": sum(value == "ventral" for value in predictions.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
