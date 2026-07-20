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
        ref = MACDReference(fillna=True, close=pd.Series(data))
        for x, y in zip([macd.update(v).signal for v in data], ref.macd_signal()):
            self.assertAlmostEqual(x, y, places=4)

        # Multi-output fields should also agree with the ta reference MACD line.
        macd2 = MACD(fillna=True)
        for out, line in zip([macd2.update(v) for v in data], ref.macd()):
            self.assertAlmostEqual(out.macd, line, places=4)
            self.assertAlmostEqual(out.histogram, out.macd - out.signal, places=12)

if __name__ == "__main__":
    main()
