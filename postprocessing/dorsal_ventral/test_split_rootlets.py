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


if __name__ == "__main__":
    unittest.main()
