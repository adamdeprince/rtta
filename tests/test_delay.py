import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import Delay


class SMATest(TestCase):

    def test(self):
        delay = Delay(window=2, fillna=False)
        self.assertTrue(math.isnan(delay.update(1)))
        self.assertTrue(math.isnan(delay.update(2)))
        self.assertAlmostEqual(delay.update(3), 1)
        self.assertAlmostEqual(delay.update(4), 2)
        self.assertAlmostEqual(delay.update(5), 3)
        self.assertAlmostEqual(delay.update(6), 4)
        self.assertAlmostEqual(delay.update(7), 5)

    def test_one(self):
        delay = Delay(window=1, fillna=False)
        self.assertTrue(math.isnan(delay.update(1)))
        self.assertAlmostEqual(delay.update(2), 1)
        self.assertAlmostEqual(delay.update(3), 2)
        self.assertAlmostEqual(delay.update(4), 3)

    def test_one_fillna(self):
        delay = Delay(window=1, fillna=True)
        self.assertAlmostEqual(delay.update(1), 0)
        self.assertAlmostEqual(delay.update(2), 1)
        self.assertAlmostEqual(delay.update(3), 2)
        self.assertAlmostEqual(delay.update(4), 3)

if __name__ == "__main__":
    main()
