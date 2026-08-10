"""Synthetic tests for the deterministic dorsal/ventral separator."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.attachment_graph import (
    build_attachment_graph,
    optimize_attachment_graph,
)
from postprocessing.dorsal_ventral.attachment_graph_cv import (
    make_leave_one_group_out_folds,
    validate_graph,
)
from postprocessing.dorsal_ventral.cluster_mean_v5 import split_cluster_mean_v5
from postprocessing.dorsal_ventral.audit_attachment_seeds import (
    audit_attachment_seeds,
)
from postprocessing.dorsal_ventral.evaluate_partial_dorsal import (
    evaluate_partial_dorsal,
)
from postprocessing.dorsal_ventral.evaluate_cord_perturbations import (
    PERTURBATIONS,
    evaluate_cord_perturbations,
)
from postprocessing.dorsal_ventral.evaluate_session_consistency import (
    pair_metrics,
    read_manifest,
    scan_metrics,
    summarize,
)
from postprocessing.dorsal_ventral.export_attachment_islands import (
    extract_attachment_islands,
)
from postprocessing.dorsal_ventral.hybrid_v5 import combine_v5_with_fallback
from postprocessing.dorsal_ventral.stage_hybrid_v5_inference import stage
from postprocessing.dorsal_ventral.run_hybrid_v5_batch import run_batch
from postprocessing.dorsal_ventral.evaluate_attachment_review import score_rows
from postprocessing.dorsal_ventral.split_rootlets import (
    _local_surface_coordinates,
    run,
    split_rootlets,
)


def _synthetic_case() -> tuple[np.ndarray, np.ndarray]:
    shape = (41, 41, 25)
    cord = np.zeros(shape, dtype=np.uint8)
    rootlets = np.zeros(shape, dtype=np.int16)
    xx, yy = np.ogrid[: shape[0], : shape[1]]
    cord_xy = ((xx - 20) / 3.0) ** 2 + ((yy - 20) / 4.0) ** 2 <= 1
    cord[:, :, :] = cord_xy[:, :, None]

    # Level 2: four anatomically separated branches. The left pair is joined
    # distally to exercise within-component geodesic separation.
    rootlets[24:35, 15:17, 5:9] = 2  # left dorsal
    rootlets[24:35, 24:26, 5:9] = 2  # left ventral
    rootlets[33:35, 16:25, 5:9] = 2  # distal bridge
    rootlets[6:17, 15:17, 5:9] = 2   # right dorsal
    rootlets[6:17, 24:26, 5:9] = 2   # right ventral

    # Level 3: disconnected branches verify single-class seed propagation.
    rootlets[24:34, 15:17, 15:19] = 3
    rootlets[24:34, 24:26, 15:19] = 3
    rootlets[7:17, 15:17, 15:19] = 3
    rootlets[7:17, 24:26, 15:19] = 3
    return rootlets, cord


class SplitRootletsTest(unittest.TestCase):
    @staticmethod
    def _graph_row(
        attachment_id: int,
        level: int,
        side: str,
        ap: float,
        *,
        predicted_class: str = "unclear",
        expert_class: str = "",
    ) -> dict[str, str]:
        return {
            "subject": "sub-synthetic",
            "attachment_id": str(attachment_id),
            "level": str(level),
            "side": side,
            "component_id": str(attachment_id),
            "predicted_class": predicted_class,
            "median_ap_mm": str(ap),
            "minimum_cord_distance_mm": "0.8",
            "eligible_for_splitter": "true",
            "voxel_count": "8",
            "expert_class": expert_class,
        }

    def test_attachment_graph_never_uses_pseudo_labels_as_features(self) -> None:
        rows = [
            self._graph_row(1, 2, "right", -1.0, predicted_class="dorsal"),
            self._graph_row(2, 2, "right", 1.0, predicted_class="ventral"),
            self._graph_row(3, 2, "left", -0.8, predicted_class="unclear"),
            self._graph_row(4, 2, "left", 1.2, predicted_class="dorsal"),
        ]
        first = build_attachment_graph(rows, group_id="sub-synthetic")
        for row in rows:
            row["predicted_class"] = "ventral"
        second = build_attachment_graph(rows, group_id="sub-synthetic")

        self.assertEqual(first, second)
        self.assertFalse(first["provenance"]["uses_input_predicted_class"])
        self.assertNotIn("predicted_class", first["feature_names"])

    def test_graph_optimizer_enforces_order_and_smooths_singletons(self) -> None:
        rows = [
            self._graph_row(1, 2, "right", -2.0),
            self._graph_row(2, 2, "right", -1.0),
            self._graph_row(3, 2, "right", 3.0),
            self._graph_row(4, 3, "right", 0.1),
            self._graph_row(5, 4, "right", -2.2),
            self._graph_row(6, 4, "right", -1.2),
            self._graph_row(7, 4, "right", 2.8),
        ]
        graph = build_attachment_graph(rows)
        predictions = optimize_attachment_graph(
            graph, bilateral_weight=0.0, level_weight=1.5
        )

        self.assertEqual(predictions["1"], "dorsal")
        self.assertEqual(predictions["2"], "dorsal")
        self.assertEqual(predictions["3"], "ventral")
        self.assertEqual(
            predictions["4"],
            "dorsal",
            "Adjacent levels should override the weak AP sign of a singleton.",
        )

    def test_graph_cv_keeps_paired_sessions_in_one_subject_group(self) -> None:
        labeled_rows = [
            self._graph_row(1, 2, "right", -1.0, expert_class="dorsal"),
            self._graph_row(2, 2, "right", 1.0, expert_class="ventral"),
        ]
        graphs = [
            build_attachment_graph(labeled_rows, group_id="sub-01"),
            build_attachment_graph(labeled_rows, group_id="sub-01"),
            build_attachment_graph(labeled_rows, group_id="sub-02"),
        ]
        folds = make_leave_one_group_out_folds(graphs)
        first = next(fold for fold in folds if fold["held_out_group"] == "sub-01")

        self.assertEqual(first["test_graphs"], 2)
        self.assertEqual(first["test_groups"], ["sub-01"])
        self.assertNotIn("sub-01", first["train_groups"])
        self.assertTrue(first["trainable"])
        self.assertTrue(first["scorable"])

    def test_graph_cv_rejects_pseudo_label_features(self) -> None:
        rows = [
            self._graph_row(1, 2, "right", -1.0),
            self._graph_row(2, 2, "right", 1.0),
        ]
        graph = build_attachment_graph(rows)
        graph["provenance"]["uses_input_predicted_class"] = True
        with self.assertRaisesRegex(ValueError, "pseudo-label"):
            validate_graph(graph)

    def test_attachment_review_scores_abstentions_as_errors(self) -> None:
        rows = [
            {"expert_class": "dorsal", "predicted_class": "dorsal"},
            {"expert_class": "dorsal", "predicted_class": "unclear"},
            {"expert_class": "ventral", "predicted_class": "ventral"},
            {"expert_class": "ventral", "predicted_class": "dorsal"},
            {"expert_class": "unclear", "predicted_class": "ventral"},
            {"expert_class": "", "predicted_class": "ventral"},
        ]
        metrics = score_rows(rows)
        self.assertEqual(metrics["expert_reviewed"], 5)
        self.assertEqual(metrics["expert_scorable"], 4)
        self.assertEqual(metrics["expert_review_coverage"], 5 / 6)
        self.assertEqual(metrics["prediction_coverage"], 0.75)
        self.assertEqual(metrics["accuracy_with_abstentions_as_errors"], 0.5)
        self.assertEqual(metrics["selective_accuracy_non_abstained"], 2 / 3)
        self.assertEqual(metrics["balanced_accuracy_with_abstentions_as_errors"], 0.5)

    def test_support_levels_and_expected_attachment_classes(self) -> None:
        rootlets, cord = _synthetic_case()
        result = split_rootlets(rootlets, cord, (0.8, 0.8, 0.8))

        np.testing.assert_array_equal(result.dorsal + result.ventral, rootlets)
        self.assertFalse(np.any((result.dorsal > 0) & (result.ventral > 0)))
        self.assertTrue(np.all(result.dorsal[rootlets[:, :, :] > 0] >= 0))
        self.assertTrue(np.all(result.dorsal[24:28, 15:17, 5:9] == 2))
        self.assertTrue(np.all(result.ventral[24:28, 24:26, 5:9] == 2))
        self.assertTrue(np.all(result.dorsal[7:12, 15:17, 15:19] == 3))
        self.assertTrue(np.all(result.ventral[7:12, 24:26, 15:19] == 3))
        self.assertGreaterEqual(
            sum(level["geodesic_components"] for level in result.qc["levels"]), 1
        )

    def test_is_deterministic(self) -> None:
        rootlets, cord = _synthetic_case()
        first = split_rootlets(rootlets, cord, (0.8, 0.8, 0.8))
        second = split_rootlets(rootlets, cord, (0.8, 0.8, 0.8))
        np.testing.assert_array_equal(first.dorsal, second.dorsal)
        np.testing.assert_array_equal(first.ventral, second.ventral)
        np.testing.assert_array_equal(first.score, second.score)

    def test_paired_attachment_v3_uses_relative_side_boundaries(self) -> None:
        rootlets, cord = _synthetic_case()
        result = split_rootlets(
            rootlets,
            cord,
            (0.8, 0.8, 0.8),
            seed_strategy="paired_attachment",
        )

        np.testing.assert_array_equal(result.dorsal + result.ventral, rootlets)
        self.assertEqual(result.qc["method"], "paired_attachment_geodesic_v3")
        self.assertEqual(result.qc["fallback_components"], 0)
        self.assertTrue(
            all(
                all(
                    boundary is not None
                    for boundary in level["paired_boundaries_mm"].values()
                )
                for level in result.qc["levels"]
            )
        )
        self.assertTrue(np.all(result.dorsal[24:28, 15:17, 5:9] == 2))
        self.assertTrue(np.all(result.ventral[24:28, 24:26, 5:9] == 2))

    def test_graph_attachment_v4_preserves_support_and_bilateral_pairs(self) -> None:
        rootlets, cord = _synthetic_case()
        result = split_rootlets(
            rootlets,
            cord,
            (0.8, 0.8, 0.8),
            seed_strategy="graph_attachment",
        )

        np.testing.assert_array_equal(result.dorsal + result.ventral, rootlets)
        self.assertEqual(result.qc["method"], "graph_attachment_geodesic_v4")
        self.assertEqual(result.qc["fallback_components"], 0)
        self.assertTrue(
            all(
                set(level["paired_boundary_sources"].values()) == {"graph"}
                for level in result.qc["levels"]
            )
        )
        self.assertTrue(np.all(result.dorsal[24:28, 15:17, 5:9] == 2))
        self.assertTrue(np.all(result.ventral[24:28, 24:26, 5:9] == 2))

    def test_cluster_mean_v5_labels_four_separated_components(self) -> None:
        rootlets = np.zeros((24, 40, 8), dtype=np.int16)
        rootlets[2:6, 7:9, 2:5] = 2
        rootlets[17:21, 9:11, 2:5] = 2
        rootlets[2:6, 27:29, 2:5] = 2
        rootlets[17:21, 29:31, 2:5] = 2

        result = split_cluster_mean_v5(
            rootlets,
            0.8,
            min_component_voxels=8,
            min_gap_mm=2.0,
            min_gap_ratio=2.0,
        )

        np.testing.assert_array_equal(
            result.dorsal + result.ventral + result.fallback, rootlets
        )
        self.assertFalse(np.any(result.fallback))
        self.assertTrue(np.all(result.ventral[2:6, 7:9, 2:5] == 2))
        self.assertTrue(np.all(result.ventral[17:21, 9:11, 2:5] == 2))
        self.assertTrue(np.all(result.dorsal[2:6, 27:29, 2:5] == 2))
        self.assertTrue(np.all(result.dorsal[17:21, 29:31, 2:5] == 2))
        self.assertEqual(result.qc["deterministic_levels"], 1)

    def test_cluster_mean_v5_routes_weak_or_singleton_levels(self) -> None:
        rootlets = np.zeros((24, 40, 8), dtype=np.int16)
        rootlets[2:6, 10:12, 2:5] = 2
        rootlets[17:21, 12:14, 2:5] = 2
        rootlets[8:12, 24:27, 5:7] = 3

        result = split_cluster_mean_v5(
            rootlets,
            0.8,
            min_component_voxels=8,
            min_gap_mm=3.0,
        )

        self.assertTrue(np.all(result.fallback[rootlets > 0] > 0))
        self.assertEqual(result.qc["deterministic_levels"], 0)
        self.assertEqual(result.qc["fallback_levels"], 2)

    def test_cluster_mean_v5_routes_a_component_crossing_the_boundary(self) -> None:
        rootlets = np.zeros((30, 50, 8), dtype=np.int16)
        rootlets[2:5, 7:10, 2:5] = 2
        rootlets[12:15, 8:33, 2:5] = 2
        rootlets[22:25, 30:33, 2:5] = 2

        result = split_cluster_mean_v5(
            rootlets,
            0.8,
            min_component_voxels=8,
            min_gap_mm=2.0,
            min_gap_ratio=1.0,
            crossing_fraction=0.1,
        )

        self.assertTrue(np.all(result.fallback[rootlets > 0] == 2))
        self.assertEqual(
            result.qc["levels"][0]["decision"], "boundary_crossing_component"
        )

    def test_hybrid_v5_fills_only_routed_support(self) -> None:
        rootlets = np.zeros((24, 40, 8), dtype=np.int16)
        rootlets[2:6, 7:9, 2:5] = 2
        rootlets[17:21, 9:11, 2:5] = 2
        rootlets[2:6, 27:29, 2:5] = 2
        rootlets[17:21, 29:31, 2:5] = 2
        rootlets[8:12, 18:22, 5:7] = 3
        fallback = np.zeros_like(rootlets, dtype=np.uint8)
        fallback[rootlets == 3] = 1

        result = combine_v5_with_fallback(
            rootlets,
            fallback,
            0.8,
            v5_config={
                "min_component_voxels": 8,
                "min_gap_mm": 2.0,
                "min_gap_ratio": 2.0,
            },
        )

        np.testing.assert_array_equal(result.dorsal + result.ventral, rootlets)
        self.assertTrue(np.all(result.dorsal[rootlets == 3] == 3))
        self.assertTrue(np.all(result.ventral[2:6, 7:9, 2:5] == 2))

    def test_hybrid_v5_recovers_background_from_dv_probabilities(self) -> None:
        rootlets = np.zeros((16, 20, 5), dtype=np.int16)
        rootlets[4:8, 8:12, 1:4] = 2
        fallback = np.zeros_like(rootlets, dtype=np.uint8)
        probabilities = np.zeros((3, *rootlets.shape), dtype=np.float32)
        probabilities[2, rootlets > 0] = 0.8
        probabilities[1, rootlets > 0] = 0.2

        result = combine_v5_with_fallback(
            rootlets,
            fallback,
            1.0,
            fallback_probabilities_rpi=probabilities,
        )

        self.assertTrue(np.all(result.ventral[rootlets > 0] == 2))
        self.assertEqual(
            result.qc["fallback_background_voxels_recovered_from_probabilities"],
            int(np.count_nonzero(rootlets)),
        )

    def test_cord_perturbation_audit_reports_fixed_support_flip_rates(self) -> None:
        rootlets, cord = _synthetic_case()
        metrics = evaluate_cord_perturbations(
            rootlets,
            cord,
            (0.8, 0.8, 0.8),
            affine=np.diag([0.8, 0.8, 0.8, 1.0]),
            seed_strategy="paired_attachment",
        )

        self.assertEqual(set(metrics["perturbations"]), set(PERTURBATIONS))
        self.assertEqual(metrics["seed_strategy"], "paired_attachment")
        self.assertEqual(metrics["paired_min_span_mm"], 0.5)
        self.assertGreaterEqual(metrics["maximum_flip_fraction"], 0.0)
        self.assertLessEqual(metrics["maximum_flip_fraction"], 1.0)
        for perturbation in metrics["perturbations"].values():
            self.assertEqual(
                perturbation["support_voxels"], int(np.count_nonzero(rootlets))
            )
            self.assertIn("2", perturbation["levels"])
            self.assertIn("3", perturbation["levels"])

    def test_cli_round_trip_preserves_non_ras_grid(self) -> None:
        rootlets, cord = _synthetic_case()
        affine = np.array(
            [[-0.8, 0, 0, 16], [0, 0.8, 0, -16], [0, 0, 0.8, -10], [0, 0, 0, 1]],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rootlet_path = directory / "rootlets.nii.gz"
            cord_path = directory / "cord.nii.gz"
            dorsal_path = directory / "dorsal.nii.gz"
            ventral_path = directory / "ventral.nii.gz"
            score_path = directory / "score.nii.gz"
            qc_path = directory / "qc.json"
            nib.save(nib.Nifti1Image(rootlets, affine), rootlet_path)
            nib.save(nib.Nifti1Image(cord, affine), cord_path)

            qc = run(
                argparse.Namespace(
                    rootlets=str(rootlet_path),
                    cord=str(cord_path),
                    output_dorsal=str(dorsal_path),
                    output_ventral=str(ventral_path),
                    output_score=str(score_path),
                    qc_json=str(qc_path),
                    seed_strategy="attachment_island",
                    attachment_distance_mm=4.0,
                    attachment_band_mm=0.8,
                    ap_margin_mm=0.5,
                )
            )

            dorsal_image = nib.load(dorsal_path)
            ventral_image = nib.load(ventral_path)
            np.testing.assert_allclose(dorsal_image.affine, affine)
            np.testing.assert_allclose(ventral_image.affine, affine)
            output_sum = np.asanyarray(dorsal_image.dataobj) + np.asanyarray(
                ventral_image.dataobj
            )
            np.testing.assert_array_equal(output_sum, rootlets)
            self.assertEqual(qc["input_voxels"], int(np.count_nonzero(rootlets)))
            self.assertTrue(qc_path.exists())

    def test_rejects_non_integer_labels(self) -> None:
        rootlets, cord = _synthetic_case()
        rootlets = rootlets.astype(float)
        rootlets[0, 0, 0] = 0.25
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            split_rootlets(rootlets, cord, (0.8, 0.8, 0.8))

    def test_rejects_negative_paired_span(self) -> None:
        rootlets, cord = _synthetic_case()
        with self.assertRaisesRegex(ValueError, "paired_min_span_mm"):
            split_rootlets(
                rootlets,
                cord,
                (0.8, 0.8, 0.8),
                seed_strategy="paired_attachment",
                paired_min_span_mm=-0.1,
            )

    def test_partial_dorsal_metrics_do_not_invent_ventral_accuracy(self) -> None:
        combined = np.array([2, 2, 2, 2, 0], dtype=np.int16)
        predicted_dorsal = np.array([2, 0, 2, 0, 0], dtype=np.int16)
        predicted_ventral = np.array([0, 2, 0, 2, 0], dtype=np.int16)
        dorsal_reference = np.array([2, 2, 0, 0, 2], dtype=np.int16)

        metrics = evaluate_partial_dorsal(
            combined, predicted_dorsal, predicted_ventral, dorsal_reference
        )

        self.assertAlmostEqual(metrics["reference_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["known_dorsal_recall"], 1 / 2)
        self.assertAlmostEqual(metrics["known_dorsal_leakage_to_ventral"], 1 / 2)
        self.assertAlmostEqual(
            metrics["known_dorsal_overlap_fraction_of_predicted_dorsal"], 1 / 2
        )
        self.assertAlmostEqual(metrics["all_dorsal_overlap_fraction_baseline"], 1 / 2)
        self.assertNotIn("ventral_accuracy", metrics)

        all_dorsal = evaluate_partial_dorsal(
            combined, combined, np.zeros_like(combined), dorsal_reference
        )
        self.assertEqual(all_dorsal["known_dorsal_recall"], 1.0)
        self.assertEqual(all_dorsal["predicted_dorsal_fraction"], 1.0)
        self.assertEqual(
            all_dorsal["known_dorsal_overlap_fraction_of_predicted_dorsal"],
            all_dorsal["all_dorsal_overlap_fraction_baseline"],
        )

    def test_partial_evaluator_rejects_interpolated_labels(self) -> None:
        combined = np.array([0.0, 2.5])
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            evaluate_partial_dorsal(
                combined,
                np.array([0, 2]),
                np.array([0, 0]),
                np.array([0, 2]),
            )

    def test_surface_coordinates_use_world_ras_for_oblique_axial_grid(self) -> None:
        angle = np.deg2rad(30)
        affine = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        nearest = np.array([[[[7]]], [[[5]]], [[[2]]]])
        centre_x = np.full(5, 5.0)
        centre_y = np.full(5, 5.0)

        _, surface_ap = _local_surface_coordinates(
            nearest, centre_x, centre_y, affine
        )

        self.assertGreater(float(surface_ap[0, 0, 0]), 0.9)

    def test_attachment_island_does_not_bisect_one_thick_branch(self) -> None:
        shape = (41, 41, 12)
        cord = np.zeros(shape, dtype=np.uint8)
        rootlets = np.zeros(shape, dtype=np.int16)
        xx, yy = np.ogrid[: shape[0], : shape[1]]
        cord_xy = ((xx - 20) / 3.0) ** 2 + ((yy - 20) / 4.0) ** 2 <= 1
        cord[:, :, :] = cord_xy[:, :, None]
        rootlets[24:35, 12:19, 4:8] = 2
        rootlets[24:25, 19:24, 4:8] = 2
        rootlets[25:35, 19:20, 4:8] = 2

        island = split_rootlets(rootlets, cord, (0.8, 0.8, 0.8))
        dense = split_rootlets(
            rootlets,
            cord,
            (0.8, 0.8, 0.8),
            seed_strategy="dense_voxel",
        )

        self.assertEqual(np.count_nonzero(island.ventral), 0)
        self.assertGreater(np.count_nonzero(dense.ventral), 0)
        np.testing.assert_array_equal(island.dorsal, rootlets)

    def test_unregistered_session_metrics_compare_proportions_not_dice(self) -> None:
        affine = np.diag([0.8, 0.8, 0.8, 1.0])
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest_path = directory / "manifest.csv"
            manifest_rows = []
            for session, dorsal_count in (("ses-01", 2), ("ses-02", 3)):
                combined = np.zeros((4, 2, 1), dtype=np.int16)
                combined[:2, :, :] = 2
                combined[2:, :, :] = 3
                dorsal = np.zeros_like(combined)
                dorsal.flat[:dorsal_count] = combined.flat[:dorsal_count]
                ventral = combined - dorsal
                prefix = directory / session
                combined_path = prefix.with_name(f"{session}_combined.nii.gz")
                dorsal_path = prefix.with_name(f"{session}_dorsal.nii.gz")
                ventral_path = prefix.with_name(f"{session}_ventral.nii.gz")
                qc_path = prefix.with_name(f"{session}_qc.json")
                nib.save(nib.Nifti1Image(combined, affine), combined_path)
                nib.save(nib.Nifti1Image(dorsal, affine), dorsal_path)
                nib.save(nib.Nifti1Image(ventral, affine), ventral_path)
                qc_path.write_text(
                    json.dumps(
                        {
                            "fallback_components": 1,
                            "levels": [
                                {
                                    "label": 2,
                                    "components": 2,
                                    "fallback_components": 1,
                                    "attachment_islands": 2,
                                    "neutral_attachment_islands": 1,
                                },
                                {
                                    "label": 3,
                                    "components": 2,
                                    "fallback_components": 0,
                                    "attachment_islands": 2,
                                    "neutral_attachment_islands": 0,
                                },
                            ],
                        }
                    )
                )
                manifest_rows.append(
                    {
                        "subject": "sub-01",
                        "session": session,
                        "combined": combined_path.name,
                        "dorsal": dorsal_path.name,
                        "ventral": ventral_path.name,
                        "qc": qc_path.name,
                    }
                )
            with manifest_path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("subject", "session", "combined", "dorsal", "ventral", "qc"),
                )
                writer.writeheader()
                writer.writerows(manifest_rows)

            rows = read_manifest(manifest_path)
            scans = [record for row in rows for record in scan_metrics(row)]
            paired, incomplete = pair_metrics(scans)
            summary = summarize(paired, incomplete)

            level_two = next(record for record in paired if record["level"] == "2")
            self.assertAlmostEqual(level_two["abs_dorsal_fraction_difference"], 0.25)
            self.assertTrue(level_two["presence_agreement"])
            self.assertEqual(summary["complete_subject_pairs"], 1)
            self.assertEqual(
                summary["highest_abs_dorsal_fraction_differences"][0]["subject"],
                "sub-01",
            )
            self.assertNotIn("dice", json.dumps(summary).lower())

    def test_session_metrics_reject_non_partition(self) -> None:
        affine = np.eye(4)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            combined_path = directory / "combined.nii.gz"
            dorsal_path = directory / "dorsal.nii.gz"
            ventral_path = directory / "ventral.nii.gz"
            qc_path = directory / "qc.json"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), affine), combined_path)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), affine), dorsal_path)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), affine), ventral_path)
            qc_path.write_text('{"levels": []}\n')
            with self.assertRaisesRegex(ValueError, "overlap"):
                scan_metrics(
                    {
                        "subject": "sub-01",
                        "session": "ses-01",
                        "combined": str(combined_path),
                        "dorsal": str(dorsal_path),
                        "ventral": str(ventral_path),
                        "qc": str(qc_path),
                    }
                )

    def test_seed_audit_identifies_known_dorsal_proximal_seeds(self) -> None:
        rootlets, cord = _synthetic_case()
        dorsal_reference = np.zeros_like(rootlets)
        dorsal_reference[24:35, 15:17, 5:9] = 2
        dorsal_reference[6:17, 15:17, 5:9] = 2
        dorsal_reference[24:34, 15:17, 15:19] = 3
        dorsal_reference[7:17, 15:17, 15:19] = 3

        metrics = audit_attachment_seeds(
            rootlets, cord, dorsal_reference, (0.8, 0.8, 0.8)
        )

        self.assertGreater(metrics["known_dorsal_proximal_voxels"], 0)
        self.assertEqual(metrics["known_dorsal_ventral_seed_rate"], 0.0)
        self.assertEqual(metrics["known_dorsal_dorsal_seed_rate"], 1.0)

    def test_attachment_export_produces_reviewable_ids(self) -> None:
        rootlets, cord = _synthetic_case()
        island_map, records = extract_attachment_islands(
            rootlets, cord, (0.8, 0.8, 0.8), subject="synthetic"
        )

        ids = np.unique(island_map[island_map > 0])
        np.testing.assert_array_equal(ids, np.arange(1, len(records) + 1))
        self.assertEqual({record["subject"] for record in records}, {"synthetic"})
        self.assertTrue(
            {record["predicted_class"] for record in records}
            <= {"dorsal", "ventral", "unclear"}
        )
        self.assertTrue(all(record["expert_class"] == "" for record in records))

    def test_hybrid_inference_stage_reorients_pairs_to_rpi(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            image_path = directory / "image.nii.gz"
            rootlet_path = directory / "rootlets.nii.gz"
            manifest_path = directory / "manifest.csv"
            output_directory = directory / "staged"
            affine = np.diag((-0.8, 0.9, 1.2, 1.0))
            anatomy = np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8)
            rootlets = np.zeros_like(anatomy, dtype=np.int16)
            rootlets[2:5, 3:6, 2:5] = 4
            nib.save(nib.Nifti1Image(anatomy, affine), image_path)
            nib.save(nib.Nifti1Image(rootlets, affine), rootlet_path)
            with manifest_path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("case_id", "dataset", "image", "rootlets")
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "datasetA_case01",
                        "dataset": "Dataset A",
                        "image": image_path.name,
                        "rootlets": rootlet_path.name,
                    }
                )

            audit = stage(manifest_path, output_directory)
            staged_image = nib.load(output_directory / "datasetA_case01_0000.nii.gz")
            staged_rootlets = nib.load(output_directory / "datasetA_case01_0001.nii.gz")

            self.assertEqual(tuple(nib.aff2axcodes(staged_image.affine)), ("R", "P", "I"))
            self.assertEqual(tuple(nib.aff2axcodes(staged_rootlets.affine)), ("R", "P", "I"))
            self.assertEqual(staged_image.shape, staged_rootlets.shape)
            self.assertEqual(audit["case_count"], 1)
            self.assertEqual(audit["cases"][0]["rootlet_voxels"], 27)
            self.assertEqual(audit["cases"][0]["fallback_levels"], 1)

    def test_hybrid_batch_preserves_staged_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            stage_directory = directory / "stage"
            prediction_directory = directory / "predictions"
            output_directory = directory / "hybrid"
            stage_directory.mkdir()
            prediction_directory.mkdir()
            shape = (8, 9, 7)
            affine = np.diag((0.8, -0.9, -1.2, 1.0))
            anatomy = np.ones(shape, dtype=np.float32)
            rootlets = np.zeros(shape, dtype=np.int16)
            rootlets[2:5, 3:6, 2:5] = 4
            nib.save(
                nib.Nifti1Image(anatomy, affine),
                stage_directory / "case01_0000.nii.gz",
            )
            rootlet_image = nib.Nifti1Image(rootlets, affine)
            nib.save(rootlet_image, stage_directory / "case01_0001.nii.gz")
            stage_manifest = {
                "schema": "rootlet-dv-hybrid-v5-inference-stage-v1",
                "orientation": "RPI",
                "cases": [
                    {
                        "case_id": "case01",
                        "dataset": "synthetic",
                        "rpi_image": str(stage_directory / "case01_0000.nii.gz"),
                        "rpi_rootlets": str(stage_directory / "case01_0001.nii.gz"),
                        "v5_level_coverage": 0.0,
                    }
                ],
            }
            (stage_directory / "stage_manifest.json").write_text(
                json.dumps(stage_manifest)
            )
            segmentation = np.zeros(shape, dtype=np.uint8)
            nib.save(
                nib.Nifti1Image(segmentation, affine),
                prediction_directory / "case01.nii.gz",
            )
            probabilities = np.zeros((3, *shape), dtype=np.float32)
            probabilities[2, rootlets > 0] = 1.0
            np.savez_compressed(
                prediction_directory / "case01.npz", probabilities=probabilities
            )

            summary = run_batch(
                stage_directory, prediction_directory, output_directory
            )
            dorsal = np.rint(
                np.asanyarray(
                    nib.load(
                        output_directory
                        / "case01_desc-dorsal_label-rootlets_dseg.nii.gz"
                    ).dataobj
                )
            ).astype(np.int16)
            ventral = np.rint(
                np.asanyarray(
                    nib.load(
                        output_directory
                        / "case01_desc-ventral_label-rootlets_dseg.nii.gz"
                    ).dataobj
                )
            ).astype(np.int16)

            self.assertEqual(summary["exact_support_partitions"], 1)
            np.testing.assert_array_equal(dorsal + ventral, rootlets)
            self.assertEqual(np.count_nonzero(dorsal), 0)
            self.assertEqual(np.count_nonzero(ventral), 27)


if __name__ == "__main__":
    unittest.main()
