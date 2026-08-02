#!/usr/bin/env python3
"""Create leakage-safe subject-group folds for attachment graph models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _label_counts(graphs: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        node["expert_label"]
        for graph in graphs
        for node in graph["nodes"]
        if node["expert_label"] in {0, 1}
    )
    return {"dorsal": counts[0], "ventral": counts[1]}


def validate_graph(graph: dict[str, Any]) -> None:
    if graph.get("schema") != "rootlet-attachment-graph-v1":
        raise ValueError("Unsupported attachment graph schema.")
    provenance = graph.get("provenance", {})
    if provenance.get("uses_input_predicted_class") is not False:
        raise ValueError("Graph may contain deterministic pseudo-label features.")
    if not graph.get("group_id"):
        raise ValueError("Every graph needs a non-empty group_id.")
    for node in graph.get("nodes", []):
        if node.get("expert_label") not in {-1, 0, 1}:
            raise ValueError("expert_label must be -1, 0, or 1.")


def make_leave_one_group_out_folds(
    graphs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    graphs = list(graphs)
    if not graphs:
        raise ValueError("At least one graph is required.")
    for graph in graphs:
        validate_graph(graph)
    group_ids = sorted({graph["group_id"] for graph in graphs})
    if len(group_ids) < 2:
        raise ValueError("At least two subject groups are required for evaluation.")

    folds: list[dict[str, Any]] = []
    for held_out in group_ids:
        train = [graph for graph in graphs if graph["group_id"] != held_out]
        test = [graph for graph in graphs if graph["group_id"] == held_out]
        train_groups = sorted({graph["group_id"] for graph in train})
        test_groups = sorted({graph["group_id"] for graph in test})
        if set(train_groups) & set(test_groups):
            raise RuntimeError("Internal error: subject-group leakage detected.")
        train_counts = _label_counts(train)
        test_counts = _label_counts(test)
        folds.append(
            {
                "held_out_group": held_out,
                "train_groups": train_groups,
                "test_groups": test_groups,
                "train_graphs": len(train),
                "test_graphs": len(test),
                "train_expert_labels": train_counts,
                "test_expert_labels": test_counts,
                "trainable": all(train_counts.values()),
                "scorable": bool(sum(test_counts.values())),
            }
        )
    return folds


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    graph_paths = [Path(path) for path in args.graph]
    graphs = [json.loads(path.read_text()) for path in graph_paths]
    folds = make_leave_one_group_out_folds(graphs)
    result = {
        "evaluation": "leave-one-subject-group-out",
        "graphs": len(graphs),
        "groups": len({graph["group_id"] for graph in graphs}),
        "expert_labels": _label_counts(graphs),
        "folds": folds,
        "limitations": [
            "Only expert dorsal/ventral labels are targets.",
            "Feature normalization and model fitting must occur inside each training fold.",
            "Deterministic predictions must not be substituted for missing expert labels.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("graphs", "groups", "expert_labels")}))


if __name__ == "__main__":
    main()
