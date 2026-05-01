from unittest import TestCase, main

import numpy as np

from rtta.indicator import Ichimoku


class IchimokuTest(TestCase):
    def test_batch_matches_update(self):
        high = np.array([11.0, 13.0, 12.0, 16.0, 17.0])
        low = np.array([9.0, 10.0, 10.0, 13.0, 14.0])

        indicator = Ichimoku(window1=2, window2=3, window3=4)
        incremental = [indicator.update(h, l) for h, l in zip(high, low)]
        batch = Ichimoku(window1=2, window2=3, window3=4).batch(high, low)

        for key in ("conversion", "base", "span_a", "span_b"):
            np.testing.assert_allclose(getattr(batch, key), [getattr(x, key) for x in incremental])


if __name__ == "__main__":
    main()
