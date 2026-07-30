#!/usr/bin/env python3
"""Tests for cross-validation metric consolidation helpers."""

import math
import unittest
from pathlib import Path

from training.cervical_cropped.summarize_cv_metrics import (
    case_id,
    contrast,
    parse_fold_summaries,
    percentile,
)


class CrossValidationSummaryTest(unittest.TestCase):
    def test_case_id(self):
        self.assertEqual(case_id("/tmp/sub-example.nii.gz"), "sub-example")

    def test_contrast_groups(self):
        self.assertEqual(contrast("sub-20_inv-1_part-mag_MP2RAGE"), "INV1")
        self.assertEqual(contrast("sub-20_inv-2_part-mag_MP2RAGE"), "INV2")
        self.assertEqual(contrast("sub-20_UNIT1"), "UNIT1")
        self.assertEqual(contrast("sub-example_000"), "T2w")

    def test_linear_percentile(self):
        self.assertEqual(percentile([0.0, 1.0], 0.25), 0.25)
        self.assertTrue(math.isnan(percentile([], 0.25)))

    def test_rejects_duplicate_fold_arguments(self):
        existing = Path(__file__).resolve()
        with self.assertRaisesRegex(ValueError, "more than once"):
            parse_fold_summaries([f"0={existing}", f"0={existing}"])


if __name__ == "__main__":
    unittest.main()
