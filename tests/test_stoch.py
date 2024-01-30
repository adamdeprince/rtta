import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import StochRSI


class SMATest(TestCase):

    def test_na(self):
        high = StochRSI(window=2, fillna=False)
        self.assertTrue(math.isnan(high.update(1)))
        self.assertTrue(math.isnan(high.update(2)))
        self.assertAlmostEqual(high.update(1), 0.03)
        self.assertAlmostEqual(high.update(2), 0)
        self.assertAlmostEqual(high.update(3), 0)
        self.assertAlmostEqual(high.update(1), 0.045)
        self.assertAlmostEqual(high.update(2), 0)

    def test(self):
        high = StochRSI(window=2, fillna=True)
        self.assertAlmostEqual(high.update(1), 0)
        self.assertAlmostEqual(high.update(3), 0.04)
        self.assertAlmostEqual(high.update(2), 0.05)
        self.assertAlmostEqual(high.update(1), 0.1375)


if __name__ == "__main__":
    main()
