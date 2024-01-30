import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import StdDev


class SMATest(TestCase):

    def test_na(self):
        stddev = StdDev(window=2, fillna=False)
        self.assertTrue(math.isnan(stddev.update(1)))
        self.assertTrue(math.isnan(stddev.update(2)))
        self.assertAlmostEqual(stddev.update(1), 0.5)
        self.assertAlmostEqual(stddev.update(2), 0.5)
        self.assertAlmostEqual(stddev.update(3), 0.5)
        self.assertAlmostEqual(stddev.update(1), 1)
        self.assertAlmostEqual(stddev.update(2), 0.5)

    def test(self):
        stddev = StdDev(window=2, fillna=True)
        self.assertAlmostEqual(stddev.update(1), 0)
        self.assertAlmostEqual(stddev.update(2), 0.5)
        self.assertAlmostEqual(stddev.update(0), 1)
        self.assertAlmostEqual(stddev.update(1), 0.5)


if __name__ == "__main__":
    main()
