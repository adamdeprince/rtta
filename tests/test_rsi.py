import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import RSI


class SMATest(TestCase):

    def test_up(self):
        rsi = RSI(window=2, fillna=True)
        self.assertEqual(rsi.update(10), 50)
        self.assertEqual(rsi.update(11), 100)

    def test_down(self):
        rsi = RSI(window=2, fillna=True)
        self.assertEqual(rsi.update(10), 50)
        self.assertEqual(rsi.update(9), 0)

    def test_long(self):
        rsi = RSI(window=2, fillna=True)
        for x in range(20):
            rsi.update(22-x)
        self.assertEqual(rsi.update(1), 0)
        self.assertEqual(rsi.update(2), 25)
        self.assertEqual(rsi.update(1), 28.57142857142857)
        self.assertEqual(rsi.update(2), 37.5)

if __name__ == "__main__":
    main()
