import math
from unittest import TestCase, main

import numpy as np

from rtta.trend import EMAIndicator 


class EMAIndicatorTest(TestCase):

    def test(self):
        ema = EMAIndicator(window=4, fillna=False)
        self.assertTrue(math.isnan(ema.update(1)))
        self.assertTrue(math.isnan(ema.update(1)))
        self.assertTrue(math.isnan(ema.update(1)))
        self.assertAlmostEqual(ema.update(1),0.87040)
        self.assertAlmostEqual(ema.update(1),0.922240)
        self.assertAlmostEqual(ema.update(1),0.953344)
        self.assertAlmostEqual(ema.update(1),0.9720064)

    def test_fillna(self):
        ema = EMAIndicator(window=4, fillna=True)
        for _ in range(10):
            ema.update(1)

        raise Exception()

    def test_fillna(self):
        ema = EMAIndicator(window=4, fillna=True)
        self.assertAlmostEqual(ema.update(1), 0.4)
        self.assertAlmostEqual(ema.update(2), 1.04)
        self.assertAlmostEqual(ema.update(3), 1.824)
        self.assertAlmostEqual(ema.update(4), 2.6944)
        self.assertAlmostEqual(ema.update(5), 3.61664)
        for _ in range(4):
            ema.update(5)
        self.assertAlmostEqual(ema.update(5), 4.8924299264)
        for _ in range(15):
            ema.update(5)
        self.assertAlmostEqual(ema.update(5), 4.999969653299962)

if __name__ == "__main__":
    main()
