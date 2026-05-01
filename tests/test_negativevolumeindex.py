from unittest import TestCase, main

import numpy as np

from rtta.indicator import NegativeVolumeIndex


class NegativeVolumeIndexTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0])
        volume = np.array([100.0, 80.0, 120.0, 90.0])

        indicator = NegativeVolumeIndex()
        incremental = [indicator.update(c, v) for c, v in zip(close, volume)]
        batch = NegativeVolumeIndex().batch(close, volume)

        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
