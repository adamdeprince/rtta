import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import High


class SMATest(TestCase):

    def test_na(self):
        high = High(window=2, fillna=False)
        self.assertTrue(math.isnan(high.update(1)))
        self.assertTrue(math.isnan(high.update(2)))
        self.assertAlmostEqual(high.update(1), 2)
        self.assertAlmostEqual(high.update(2), 2)
        self.assertAlmostEqual(high.update(3), 3)
        self.assertAlmostEqual(high.update(1), 3)
        self.assertAlmostEqual(high.update(2), 2)

    def test(self):
        high = High(window=2, fillna=True)
        self.assertAlmostEqual(high.update(1), 1)
        self.assertAlmostEqual(high.update(2), 2)
        self.assertAlmostEqual(high.update(0), 2)
        self.assertAlmostEqual(high.update(1), 1)


if __name__ == "__main__":
    main()
