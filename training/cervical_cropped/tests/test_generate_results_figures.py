#!/usr/bin/env python3
"""Regression tests for cropped-rootlets result figure selection."""

import unittest

from training.cervical_cropped.generate_results_figures import level_is_present


class LevelInventoryTest(unittest.TestCase):
    def test_exact_semicolon_delimited_match(self):
        self.assertTrue(level_is_present("C2;C8;T1", "T1"))
        self.assertFalse(level_is_present("C2;C8", "T1"))

    def test_does_not_use_substring_matches(self):
        self.assertFalse(level_is_present("T10", "T1"))

    def test_empty_inventory(self):
        self.assertFalse(level_is_present("", "T1"))
        self.assertFalse(level_is_present(None, "T1"))


if __name__ == "__main__":
    unittest.main()
