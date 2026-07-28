#!/usr/bin/env python3
"""Tests for five-fold coverage validation."""

import unittest

from training.cervical_cropped.create_splits import validate_cross_validation_coverage


class CrossValidationCoverageTest(unittest.TestCase):
    def test_exact_partition(self):
        splits = [
            {"train": ["b"], "val": ["a"]},
            {"train": ["a"], "val": ["b"]},
        ]
        validate_cross_validation_coverage(splits, {"a", "b"})

    def test_rejects_duplicate_and_missing_validation_cases(self):
        splits = [
            {"train": ["b"], "val": ["a"]},
            {"train": ["b"], "val": ["a"]},
        ]
        with self.assertRaisesRegex(
            ValueError,
            r"missing=\['b'\], duplicated=\['a'\]",
        ):
            validate_cross_validation_coverage(splits, {"a", "b"})

    def test_rejects_unexpected_validation_case(self):
        splits = [{"train": ["a"], "val": ["outside"]}]
        with self.assertRaisesRegex(ValueError, r"unexpected=\['outside'\]"):
            validate_cross_validation_coverage(splits, {"a"})


if __name__ == "__main__":
    unittest.main()
