import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import PercentagePrice
from ta.momentum import PercentagePriceOscillator as PercentagePriceOscillator

class PercentagePriceOscillatorTest(TestCase):

    def test_against_reference(self):
        data = np.random.rand(100)
        ppo = PercentagePrice(fillna=True)
        for (x,y) in zip(PercentagePriceOscillator(fillna=True, close=pd.Series(data)).ppo(),
                         [ppo.update(x)['ppo'] for x in data]):
            self.assertAlmostEqual(x,y)
        ppo = PercentagePrice(fillna=True)
        for (x,y) in zip(PercentagePriceOscillator(fillna=True, close=pd.Series(data)).ppo_signal(),
                         [ppo.update(x)['signal'] for x in data]):
            self.assertAlmostEqual(x,y)
        ppo = PercentagePrice(fillna=True)
        for (x,y) in zip(PercentagePriceOscillator(fillna=True, close=pd.Series(data)).ppo_hist(),
                         [ppo.update(x)['histogram'] for x in data]):
            self.assertAlmostEqual(x,y)

        


if __name__ == "__main__":
    main()
