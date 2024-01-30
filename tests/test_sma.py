import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import SMA 
from ta.trend import SMAIndicator as SMAReference

class SMATest(TestCase):

    def test(self):
        sma = SMA(window=4, fillna=False)
        self.assertTrue(math.isnan(sma.update(1)))
        self.assertTrue(math.isnan(sma.update(2)))
        self.assertTrue(math.isnan(sma.update(3)))
        self.assertAlmostEqual(sma.update(4), sum([1,2,3,4])/4)
        self.assertAlmostEqual(sma.update(5), sum([2,3,4, 5]) / 4)

    def test_fillna(self):
        sma = SMA(window=4, fillna=True)
        self.assertAlmostEqual(sma.update(1), sum([1])/1)
        self.assertAlmostEqual(sma.update(2), sum([1,2])/2)
        self.assertAlmostEqual(sma.update(3), sum([1,2,3])/3)
        self.assertAlmostEqual(sma.update(4), sum([1,2,3,4])/4)
        self.assertAlmostEqual(sma.update(5), sum([2,3,4, 5]) / 4)

    def test_against_reference(self):
        data = np.random.rand(100)
        sma = SMA(window=4, fillna=True)
        for (x,y) in zip(SMAReference(window=4, fillna=True, close=pd.Series(data)).sma_indicator(),
                         [sma.update(x) for x in data]):
            self.assertAlmostEqual(x,y)


if __name__ == "__main__":
    main()
