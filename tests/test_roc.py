import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import ROC
from ta.momentum import ROCIndicator

class EMAIndicatorTest(TestCase):
    def test_against_reference_batch(self):
        close = np.random.rand(100)+1
        roc = ROC(window=10, fillna=True)
        for (x,y) in zip(ROCIndicator(window=10, fillna=True, close=pd.Series(close)).roc(),
                         roc.batch(close)):
            self.assertAlmostEqual(x,y)

    def test_against_reference_batch_nan(self):
        close = np.random.rand(100)+1
        roc = ROC(window=10, fillna=False)
        for (x,y) in zip(ROCIndicator(window=10, fillna=False, close=pd.Series(close)).roc(),
                         roc.batch(close)):
            if math.isnan(x):
                self.assertTrue(math.isnan(y))
            else:
                self.assertAlmostEqual(x,y)


    def test_against_reference(self):
        close = np.random.rand(100)+1
        roc = ROC(window=10, fillna=True)
        for (x,y) in zip(ROCIndicator(window=10, fillna=True, close=pd.Series(close)).roc(),
                         [roc.update(x) for x in close]):
            self.assertAlmostEqual(x,y)

    def test_against_reference_nan(self):
        close = np.random.rand(100)+1
        roc = ROC(window=10, fillna=False)
        for (x,y) in zip(ROCIndicator(window=10, fillna=False, close=pd.Series(close)).roc(),
                         [roc.update(x) for x in close]):
            if math.isnan(x):
                self.assertTrue(math.isnan(y))
            else:
                self.assertAlmostEqual(x,y)

        

        

if __name__ == "__main__":
    main()
