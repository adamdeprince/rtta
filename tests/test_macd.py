import math
from random import random
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import MACD
from ta.trend import MACD as MACDReference


class MACDTest(TestCase):

    def test_against_random_reference(self):
        data = np.random.rand(100)

        macd = MACD(fillna=True)
        for i, (x,y) in enumerate(zip([macd.update(x) for x in data],
                                      MACDReference(fillna=True, close=pd.Series(data)).macd_signal())):
            self.assertAlmostEqual(x,y, places=4)

if __name__ == "__main__":
    main()
