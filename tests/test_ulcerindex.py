from unittest import TestCase, main

import numpy as np

from rtta.indicator import UlcerIndex


class UlcerIndexTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 9.0, 15.0])

        indicator = UlcerIndex(window=3)
        incremental = [indicator.update(x) for x in close]
        batch = UlcerIndex(window=3).batch(close)

        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
