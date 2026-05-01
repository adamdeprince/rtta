from unittest import TestCase, main

import numpy as np

from rtta.indicator import SchaffTrendCycle


class SchaffTrendCycleTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0, 17.0])

        indicator = SchaffTrendCycle(slow=4, fast=2, cycle=2, smooth1=2, smooth2=2)
        incremental = [indicator.update(x) for x in close]
        batch = SchaffTrendCycle(slow=4, fast=2, cycle=2, smooth1=2, smooth2=2).batch(close)

        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
