import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import Low


class SMATest(TestCase):

    def test_na(self):
        low = Low(window=2, fillna=False)
        self.assertTrue(math.isnan(low.update(1)))
        self.assertTrue(math.isnan(low.update(2)))
        self.assertAlmostEqual(low.update(1), 1)
        self.assertAlmostEqual(low.update(2), 1)
        self.assertAlmostEqual(low.update(3), 2)
        self.assertAlmostEqual(low.update(1), 1)
        self.assertAlmostEqual(low.update(2), 1)

    def test(self):
        low = Low(window=2, fillna=True)
        self.assertAlmostEqual(low.update(1), 1)
        self.assertAlmostEqual(low.update(2), 1)
        self.assertAlmostEqual(low.update(0), 0)
        self.assertAlmostEqual(low.update(1), 0)


if __name__ == "__main__":
    main()
