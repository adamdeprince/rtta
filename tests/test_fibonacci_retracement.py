from unittest import TestCase, main

import numpy as np

from rtta.indicator import FibonacciRetracementLevels


FIELDS = ("level0", "level236", "level382", "level500", "level618", "level100")


def reference_levels(high, low, window=30, uptrend=True, fillna=True):
    output = {field: [] for field in FIELDS}
    for index in range(len(high)):
        count = min(window, index + 1)
        if not fillna and count < window:
            for field in FIELDS:
                output[field].append(np.nan)
            continue

        start = index + 1 - count
        highest = np.max(high[start : index + 1])
        lowest = np.min(low[start : index + 1])
        span = highest - lowest
        if uptrend:
            values = (
                highest,
                highest - 0.236 * span,
                highest - 0.382 * span,
                highest - 0.5 * span,
                highest - 0.618 * span,
                lowest,
            )
        else:
            values = (
                lowest,
                lowest + 0.236 * span,
                lowest + 0.382 * span,
                lowest + 0.5 * span,
                lowest + 0.618 * span,
                highest,
            )
        for field, value in zip(FIELDS, values):
            output[field].append(value)
    return {field: np.asarray(values, dtype=np.float64) for field, values in output.items()}


class FibonacciRetracementLevelsTest(TestCase):
    def test_uptrend_matches_reference(self):
        high = np.array([11.0, 13.0, 12.0, 16.0, 15.0])
        low = np.array([9.0, 10.0, 10.0, 13.0, 12.0])
        indicator = FibonacciRetracementLevels(window=3)
        incremental = [indicator.update(h, l) for h, l in zip(high, low)]
        expected = reference_levels(high, low, window=3)

        for field in FIELDS:
            np.testing.assert_allclose([getattr(row, field) for row in incremental], expected[field])

    def test_downtrend_matches_reference(self):
        high = np.array([11.0, 13.0, 12.0, 16.0, 15.0])
        low = np.array([9.0, 10.0, 10.0, 13.0, 12.0])
        batch = FibonacciRetracementLevels(window=3, uptrend=False).batch(high, low)
        expected = reference_levels(high, low, window=3, uptrend=False)

        for field in FIELDS:
            np.testing.assert_allclose(getattr(batch, field), expected[field])

    def test_fillna_false_scalar_and_replay_outputs(self):
        high = np.array([11.0, 13.0, 12.0, 16.0, 15.0])
        low = np.array([9.0, 10.0, 10.0, 13.0, 12.0])
        expected = reference_levels(high, low, window=3, fillna=False)
        indicator = FibonacciRetracementLevels(window=3, fillna=False)
        batch = indicator.batch(high, low)

        for field in FIELDS:
            np.testing.assert_allclose(getattr(batch, field), expected[field], equal_nan=True)

        replay = FibonacciRetracementLevels(window=3, fillna=False).replay_update_outputs(high, low)
        for field in FIELDS:
            np.testing.assert_allclose(getattr(replay, field), expected[field], equal_nan=True)

        scalar = FibonacciRetracementLevels(window=3)
        result = scalar.update_level382(11.0, 9.0)
        self.assertAlmostEqual(result, scalar.last_level382())


if __name__ == "__main__":
    main()
