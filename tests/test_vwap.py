from unittest import TestCase, main

import numpy as np

from rtta.indicator import VolumeWeightedAveragePrice


class VolumeWeightedAveragePriceTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0])
        high = np.array([11.0, 13.0, 12.0, 16.0])
        low = np.array([9.0, 10.0, 10.0, 13.0])
        volume = np.array([100.0, 150.0, 120.0, 130.0])

        indicator = VolumeWeightedAveragePrice(window=2)
        incremental = [indicator.update(c, h, l, v) for c, h, l, v in zip(close, high, low, volume)]
        batch = VolumeWeightedAveragePrice(window=2).batch(close, high, low, volume)

        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
