from unittest import TestCase, main

import numpy as np

from rtta.indicator import VolumeWeightedMovingAverage


def reference_vwma(close, volume, window=20, fillna=True):
    output = []
    for index in range(len(close)):
        count = min(window, index + 1)
        if not fillna and count < window:
            output.append(np.nan)
            continue
        start = index + 1 - count
        numerator = np.sum(close[start : index + 1] * volume[start : index + 1])
        denominator = np.sum(volume[start : index + 1])
        output.append(0.0 if denominator == 0.0 else numerator / denominator)
    return np.asarray(output, dtype=np.float64)


class VolumeWeightedMovingAverageTest(TestCase):
    def test_matches_reference(self):
        close = np.array([10.0, 12.0, 11.0, 15.0, 14.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0, 90.0])
        indicator = VolumeWeightedMovingAverage(window=3)

        incremental = np.asarray([indicator.update(c, v) for c, v in zip(close, volume)])
        expected = reference_vwma(close, volume, window=3)

        np.testing.assert_allclose(incremental, expected)

    def test_fillna_false(self):
        close = np.array([10.0, 12.0, 11.0, 15.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0])
        batch = VolumeWeightedMovingAverage(window=3, fillna=False).batch(close, volume)
        expected = reference_vwma(close, volume, window=3, fillna=False)

        np.testing.assert_allclose(batch, expected, equal_nan=True)

    def test_batch_and_replay_match_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0, 14.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0, 90.0])

        indicator = VolumeWeightedMovingAverage(window=2)
        incremental = np.asarray([indicator.update(c, v) for c, v in zip(close, volume)])
        batch = VolumeWeightedMovingAverage(window=2).batch(close, volume)
        np.testing.assert_allclose(batch, incremental)

        replay = VolumeWeightedMovingAverage(window=2)
        checksum = replay.replay_update(close, volume)
        self.assertIsInstance(checksum, float)
        self.assertAlmostEqual(replay.update(16.0, 110.0), VolumeWeightedMovingAverage(window=2).batch(
            np.array([14.0, 16.0]),
            np.array([90.0, 110.0]),
        )[-1])


if __name__ == "__main__":
    main()
