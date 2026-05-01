from unittest import TestCase, main

import numpy as np

from rtta.indicator import DonchianChannel


class DonchianChannelTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0])
        high = np.array([11.0, 13.0, 12.0, 16.0])
        low = np.array([9.0, 10.0, 10.0, 13.0])

        indicator = DonchianChannel(window=2)
        incremental = [indicator.update(c, h, l) for c, h, l in zip(close, high, low)]
        batch = DonchianChannel(window=2).batch(close, high, low)

        for key in ("upper", "lower", "middle", "width", "percent"):
            np.testing.assert_allclose(getattr(batch, key), [getattr(x, key) for x in incremental])


if __name__ == "__main__":
    main()
