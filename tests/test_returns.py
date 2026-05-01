import math
from unittest import TestCase, main

import numpy as np

from rtta.indicator import CumulativeReturn, DailyLogReturn, DailyReturn


class ReturnIndicatorTest(TestCase):
    def test_daily_return(self):
        close = np.array([10.0, 12.0, 11.0])
        indicator = DailyReturn()
        incremental = [indicator.update(x) for x in close]
        batch = DailyReturn().batch(close)

        np.testing.assert_allclose(batch, incremental)
        self.assertEqual(batch[0], 0)
        self.assertEqual(batch[1], 20)

    def test_daily_log_return(self):
        close = np.array([10.0, 12.0, 11.0])
        indicator = DailyLogReturn()
        incremental = [indicator.update(x) for x in close]
        batch = DailyLogReturn().batch(close)

        np.testing.assert_allclose(batch, incremental)
        self.assertAlmostEqual(batch[1], math.log(12 / 10) * 100)

    def test_cumulative_return(self):
        close = np.array([10.0, 12.0, 11.0])
        indicator = CumulativeReturn()
        incremental = [indicator.update(x) for x in close]
        batch = CumulativeReturn().batch(close)

        np.testing.assert_allclose(batch, incremental)
        self.assertEqual(batch[0], 0)
        self.assertEqual(batch[1], 20)


if __name__ == "__main__":
    main()
