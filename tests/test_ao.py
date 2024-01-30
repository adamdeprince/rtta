import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import AwesomeOscillator
from ta.momentum import AwesomeOscillatorIndicator as AwesomeOscillatorIndicatorReference

class EMAIndicatorTest(TestCase):

    def test_against_reference(self):
        high = np.random.rand(100)+1
        low = np.random.rand(100)
        ao = AwesomeOscillator(fillna=True)
        for (x,y) in zip(AwesomeOscillatorIndicatorReference(fillna=True, high=pd.Series(high), low=pd.Series(low)).awesome_oscillator(),
                         [ao.update(x, y) for x, y in zip(high, low)]):
            self.assertAlmostEqual(x,y)


if __name__ == "__main__":
    main()
