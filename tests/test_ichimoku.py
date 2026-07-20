from unittest import TestCase, main

import numpy as np

from rtta.indicator import Ichimoku


class IchimokuTest(TestCase):
    def test_batch_matches_update(self):
        high = np.array([11.0, 13.0, 12.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([9.0, 10.0, 10.0, 13.0, 14.0, 15.0, 16.0])
        close = np.array([10.0, 12.0, 11.0, 15.0, 16.0, 17.0, 18.0])

        indicator = Ichimoku(window1=2, window2=3, window3=4)
        incremental = [indicator.update(h, l, c) for h, l, c in zip(high, low, close)]
        batch = Ichimoku(window1=2, window2=3, window3=4).batch(high, low, close)

        for key in ("conversion", "base", "span_a", "span_b", "lagging_span"):
            np.testing.assert_allclose(getattr(batch, key), [getattr(x, key) for x in incremental])

    def test_lagging_span_is_delayed_close(self):
        high = np.arange(10.0, 20.0)
        low = high - 1.0
        close = high - 0.5
        ind = Ichimoku(window1=2, window2=3, window3=4, fillna=True)
        outs = [ind.update(float(h), float(l), float(c)) for h, l, c in zip(high, low, close)]
        self.assertAlmostEqual(outs[5].lagging_span, close[2])


if __name__ == "__main__":
    main()
