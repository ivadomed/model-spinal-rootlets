import math
import unittest

from training.cervical_cropped.nnunet_custom_trainers.rootlets_early_stopping.early_stopping import (
    plateau_state,
    should_stop_early,
)


class EarlyStoppingPolicyTest(unittest.TestCase):
    def test_stops_after_patience(self):
        scores = [0.1, 0.2, 0.203, 0.2035, 0.2038, 0.2039]
        stop, state = should_stop_early(
            scores, min_epochs=4, patience=3, min_delta=0.002
        )
        self.assertTrue(stop)
        self.assertEqual(state.last_significant_improvement_epoch, 2)

    def test_accumulated_small_gains_reset_reference(self):
        state = plateau_state([0.1, 0.101, 0.1021], min_delta=0.002)
        self.assertEqual(state.last_significant_improvement_epoch, 2)
        self.assertAlmostEqual(state.significant_best, 0.1021)

    def test_warmup_prevents_early_stop(self):
        stop, _ = should_stop_early(
            [0.2] * 10, min_epochs=20, patience=3, min_delta=0.002
        )
        self.assertFalse(stop)

    def test_nonfinite_history_never_triggers_stop(self):
        stop, state = should_stop_early(
            [math.nan] * 20, min_epochs=10, patience=3, min_delta=0.002
        )
        self.assertFalse(stop)
        self.assertIsNone(state.last_significant_improvement_epoch)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            should_stop_early([], min_epochs=0, patience=1, min_delta=0)
        with self.assertRaises(ValueError):
            should_stop_early([], min_epochs=1, patience=0, min_delta=0)
        with self.assertRaises(ValueError):
            plateau_state([], min_delta=-1)


if __name__ == "__main__":
    unittest.main()
