import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import ATRP

class ATRTest(TestCase):

    def test(self):
        atr = ATRP(window=2)
        self.assertEqual(atr.update(1,2,3), 1.5)
        self.assertEqual(atr.update(1,2,4), 3.0)
        self.assertEqual(atr.update(2,3,4), 1.5)
        self.assertEqual(atr.update(2,3,4), 1.25)

if __name__ == "__main__":
    main()
