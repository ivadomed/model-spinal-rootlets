"""Tests for the frozen-test V2–V5 review guard."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from postprocessing.dorsal_ventral.render_v2_v5_review import (
    FULL_METHODS,
    _validate_cases,
)


class TestReviewValidation(unittest.TestCase):
    def setUp(self) -> None:
        support = np.asarray([[[1]], [[1]], [[0]]], dtype=np.uint8)
        expert = np.asarray([[[1]], [[2]], [[0]]], dtype=np.uint8)
        self.case = {
            "alias": "Case A",
            "rootlets": support,
            "Expert": expert,
            **{name: expert.copy() for name in FULL_METHODS},
            "V5 gate": np.asarray([[[1]], [[0]], [[0]]], dtype=np.uint8),
        }
        self.summary = {
            "cases": [
                {
                    "case": "Case A",
                    "methods": {
                        name: {"voxel_accuracy": 1.0} for name in FULL_METHODS
                    },
                }
            ],
            "v5_gate": {
                "metrics": {
                    "deterministic_voxel_coverage": 0.5,
                    "deterministic_voxel_accuracy": 1.0,
                }
            },
        }

    def test_accepts_exact_rootletseg_partitions(self) -> None:
        _validate_cases(self.summary, [self.case])

    def test_rejects_classifier_voxels_outside_rootletseg(self) -> None:
        case = copy.deepcopy(self.case)
        case["3-D classifier"][2, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "does not exactly partition RootletSeg"):
            _validate_cases(self.summary, [case])


if __name__ == "__main__":
    unittest.main()
