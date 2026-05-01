from unittest import TestCase, main

import numpy as np

from rtta.indicator import VolumePriceTrend


class VolumePriceTrendTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0])

        indicator = VolumePriceTrend()
        incremental = [indicator.update(c, v) for c, v in zip(close, volume)]
        batch = VolumePriceTrend().batch(close, volume)

        np.testing.assert_allclose(batch, incremental)

    def test_smoothed_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0])
        indicator = VolumePriceTrend(smoothing_window=2)

        incremental = [indicator.update(c, v) for c, v in zip(close, volume)]
        batch = VolumePriceTrend(smoothing_window=2).batch(close, volume)

        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
