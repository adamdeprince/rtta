import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import ATR

class ATRTest(TestCase):

    def test(self):
        atr = ATR(window=2)
        self.assertEquals(atr.update(1,2,3), 1.5)
        self.assertEquals(atr.update(1,2,4), 3.0)
        self.assertEquals(atr.update(2,3,4), 3.0)
        self.assertEquals(atr.update(2,3,4), 2.5)

if __name__ == "__main__":
    main()
