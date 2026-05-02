import math
from unittest import TestCase, main

import numpy as np

from rtta.indicator import VariableIndexDynamicAverage


def reference_vidya(close, cmo_window=9, ema_window=12, fillna=True):
    gains = []
    losses = []
    output = []
    smoothing = 2.0 / (ema_window + 1.0)
    value = None
    previous = None

    for current in close:
        if previous is None:
            gain = 0.0
            loss = 0.0
            value = float(current)
        else:
            change = float(current) - previous
            gain = max(change, 0.0)
            loss = max(-change, 0.0)

        gains.append(gain)
        losses.append(loss)
        if len(gains) > cmo_window:
            gains.pop(0)
            losses.pop(0)

        if previous is not None:
            up_sum = sum(gains)
            down_sum = sum(losses)
            denominator = up_sum + down_sum
            cmo = abs((up_sum - down_sum) / denominator) if denominator else 0.0
            alpha = smoothing * cmo
            value = float(current) * alpha + value * (1.0 - alpha)

        output.append(math.nan if not fillna and len(gains) < cmo_window else value)
        previous = float(current)

    return np.asarray(output, dtype=np.float64)


class VariableIndexDynamicAverageTest(TestCase):
    def test_update_matches_reference(self):
        close = [100.0, 101.5, 100.75, 102.0, 102.25, 101.0, 103.5]
        indicator = VariableIndexDynamicAverage(cmo_window=3, ema_window=5)
        actual = np.asarray([indicator.update(value) for value in close])
        expected = reference_vidya(close, cmo_window=3, ema_window=5)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_matches_reference_for_realistic_sequence(self):
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.8, 512))
        actual = VariableIndexDynamicAverage(cmo_window=9, ema_window=12).batch(close)
        expected = reference_vidya(close, cmo_window=9, ema_window=12)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_fillna_false(self):
        close = np.asarray([10.0, 10.5, 10.25, 10.75, 11.0])
        actual = VariableIndexDynamicAverage(cmo_window=4, ema_window=3, fillna=False).batch(close)
        expected = reference_vidya(close, cmo_window=4, ema_window=3, fillna=False)
        self.assertTrue(np.isnan(actual[:3]).all())
        np.testing.assert_allclose(actual[3:], expected[3:], rtol=1e-12, atol=1e-12)

    def test_replay_update_checksum_advances_state(self):
        close = np.asarray([20.0, 20.2, 19.8, 20.5, 20.7])
        indicator = VariableIndexDynamicAverage(cmo_window=3, ema_window=4)
        checksum = indicator.replay_update(close)
        self.assertIsInstance(checksum, float)

        expected_state = VariableIndexDynamicAverage(cmo_window=3, ema_window=4)
        for value in close:
            expected_state.update(value)
        self.assertAlmostEqual(indicator.update(21.0), expected_state.update(21.0))

    def test_replay_advance_checksum_advances_state(self):
        close = np.asarray([20.0, 20.2, 19.8, 20.5, 20.7])
        indicator = VariableIndexDynamicAverage(cmo_window=3, ema_window=4)
        checksum = indicator.replay_advance(close)
        self.assertIsInstance(checksum, float)

        expected = VariableIndexDynamicAverage(cmo_window=3, ema_window=4)
        for value in close:
            expected.update(value)
        self.assertAlmostEqual(indicator.update(21.0), expected.update(21.0))

    def test_replay_update_checksum_matches_reference_sum(self):
        close = np.asarray([20.0, 20.2, 19.8, 20.5, 20.7])
        actual = VariableIndexDynamicAverage(cmo_window=3, ema_window=4).replay_update(close)
        expected = reference_vidya(close, cmo_window=3, ema_window=4)
        self.assertAlmostEqual(actual, float(np.sum(expected)))


if __name__ == "__main__":
    main()
