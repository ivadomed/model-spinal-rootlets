"""Tests for the frozen V2-V5 comparison helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from postprocessing.dorsal_ventral.compare_v2_v5 import (
    _elapsed_seconds,
    _full_metrics,
    _timing,
)
from postprocessing.dorsal_ventral.prepare_v2_v4_cord_masks import (
    _bounds,
    _crop_affine,
)


class CompareV2V5Test(unittest.TestCase):
    def test_full_metrics_are_scoped_to_rootlet_support(self) -> None:
        rootlets = np.zeros((2, 2, 2), dtype=np.uint8)
        rootlets[0, 0, 0] = 1
        rootlets[0, 0, 1] = 1
        target = np.zeros_like(rootlets)
        target[0, 0, 0] = 1
        target[0, 0, 1] = 2
        prediction = np.full_like(rootlets, 2)
        prediction[0, 0, 0] = 1

        metrics, counts = _full_metrics(target, prediction, rootlets)

        self.assertEqual(counts["support"], 2)
        self.assertEqual(metrics["voxel_accuracy"], 1.0)
        self.assertEqual(metrics["flipped_voxel_accuracy"], 0.0)

    def test_elapsed_seconds_parses_gnu_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timing.txt"
            path.write_text("Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n")

            self.assertEqual(_elapsed_seconds(path), 62.5)

    def test_timing_reports_median_and_range(self) -> None:
        result = _timing([3.0, 1.0, 2.0])

        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["median_seconds"], 2.0)
        self.assertEqual(result["range_seconds"], [1.0, 3.0])

    def test_cord_crop_bounds_and_affine_preserve_world_coordinates(self) -> None:
        mask = np.zeros((20, 30, 40), dtype=bool)
        mask[5:7, 10:13, 20:24] = True
        bounds = _bounds(mask, (1.0, 2.0, 4.0), padding_mm=4.0)
        affine = np.diag([1.0, -2.0, -4.0, 1.0])
        cropped_affine = _crop_affine(affine, bounds)

        self.assertEqual(bounds, (slice(1, 11), slice(8, 15), slice(19, 25)))
        original_world = nib.affines.apply_affine(affine, (1, 8, 19))
        cropped_world = nib.affines.apply_affine(cropped_affine, (0, 0, 0))
        np.testing.assert_allclose(cropped_world, original_world)


if __name__ == "__main__":
    unittest.main()
