from unittest import TestCase, main

import numpy as np

from rtta.indicator import ChaikinMoneyFlow


class ChaikinMoneyFlowTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([8.0, 5.0, 7.0, 9.0])
        high = np.array([10.0, 10.0, 9.0, 10.0])
        low = np.array([5.0, 5.0, 6.0, 8.0])
        volume = np.array([100.0, 50.0, 80.0, 75.0])

        indicator = ChaikinMoneyFlow(window=2)
        incremental = [indicator.update(c, h, l, v) for c, h, l, v in zip(close, high, low, volume)]
        batch = ChaikinMoneyFlow(window=2).batch(close, high, low, volume)

        np.testing.assert_allclose(batch, incremental)


if __name__ == "__main__":
    main()
