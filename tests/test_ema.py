import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.trend import EMAIndicator 
from ta.trend import EMAIndicator as EMAIndicatorReference

class EMAIndicatorTest(TestCase):

    def test_against_reference(self):
        data = np.random.rand(100)
        ema = EMAIndicator(window=4, fillna=True)
        for (x,y) in zip(EMAIndicatorReference(window=4, fillna=True, close=pd.Series(data)).ema_indicator(),
                         [ema.update(x) for x in data]):
            self.assertAlmostEqual(x,y)

    def test_fillna_all_ones(self):
        ema = EMAIndicator(window=4, fillna=True)
        for _ in range(10):
            self.assertAlmostEqual(ema.update(1), 1)

if __name__ == "__main__":
    main()
