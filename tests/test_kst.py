from unittest import TestCase, main

import numpy as np

from rtta.indicator import KSTOscillator


class KSTOscillatorTest(TestCase):
    def test_batch_matches_update(self):
        close = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0])
        kwargs = {
            "roc1": 1,
            "roc2": 2,
            "roc3": 3,
            "roc4": 4,
            "window1": 2,
            "window2": 2,
            "window3": 2,
            "window4": 2,
            "signal": 2,
        }

        indicator = KSTOscillator(**kwargs)
        incremental = [indicator.update(x) for x in close]
        batch = KSTOscillator(**kwargs).batch(close)

        for key in ("kst", "signal", "difference"):
            np.testing.assert_allclose(getattr(batch, key), [getattr(x, key) for x in incremental])


if __name__ == "__main__":
    main()
