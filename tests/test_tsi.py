import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import TSI


class SMATest(TestCase):


    def test(self):
        tsi = TSI()
        self.assertAlmostEqual(tsi.update(1), 100)
        self.assertAlmostEqual(tsi.update(2), 100)
        self.assertAlmostEqual(tsi.update(0), 95.6521739130435)
        self.assertAlmostEqual(tsi.update(1), 92.32500296103281)


if __name__ == "__main__":
    main()
