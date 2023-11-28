import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import PercentagePriceOscillator
from ta.momentum import PercentagePriceOscillator as PercentagePriceOscillatorReference

class PercentagePriceOscillatorTest(TestCase):

    def test_against_reference(self):
        data = np.random.rand(100)
        ppo = PercentagePriceOscillator(fillna=True)
        for (x,y) in zip(PercentagePriceOscillatorReference(fillna=True, close=pd.Series(data)).ppo(),
                         [ppo.update(x)['ppo'] for x in data]):
            self.assertAlmostEqual(x,y)
        ppo = PercentagePriceOscillator(fillna=True)
        for (x,y) in zip(PercentagePriceOscillatorReference(fillna=True, close=pd.Series(data)).ppo_signal(),
                         [ppo.update(x)['signal'] for x in data]):
            self.assertAlmostEqual(x,y)
        ppo = PercentagePriceOscillator(fillna=True)
        for (x,y) in zip(PercentagePriceOscillatorReference(fillna=True, close=pd.Series(data)).ppo_hist(),
                         [ppo.update(x)['histogram'] for x in data]):
            self.assertAlmostEqual(x,y)

    def test_batch(self):
        data = np.random.rand(100)
        ppo = PercentagePriceOscillator(fillna=True)
        for (x,y) in zip(PercentagePriceOscillatorReference(fillna=True, close=pd.Series(data)).ppo(),
                         ppo.batch(data)['ppo']):
            self.assertAlmostEqual(x,y)
        ppo = PercentagePriceOscillator(fillna=True)
        for (x,y) in zip(PercentagePriceOscillatorReference(fillna=True, close=pd.Series(data)).ppo_signal(),
                         ppo.batch(data)['signal']):
            self.assertAlmostEqual(x,y)
        ppo = PercentagePriceOscillator(fillna=True)
        for (x,y) in zip(PercentagePriceOscillatorReference(fillna=True, close=pd.Series(data)).ppo_hist(),
                         ppo.batch(data)['histogram']):
            self.assertAlmostEqual(x,y)
        


if __name__ == "__main__":
    main()
