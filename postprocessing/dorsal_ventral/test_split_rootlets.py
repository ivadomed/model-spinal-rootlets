"""Synthetic tests for the deterministic dorsal/ventral separator."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.evaluate_partial_dorsal import (
    evaluate_partial_dorsal,
)
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
                    attachment_distance_mm=4.0,
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
        self.assertNotIn("ventral_accuracy", metrics)

        all_dorsal = evaluate_partial_dorsal(
            combined, combined, np.zeros_like(combined), dorsal_reference
        )
        self.assertEqual(all_dorsal["known_dorsal_recall"], 1.0)
        self.assertEqual(all_dorsal["predicted_dorsal_fraction"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
