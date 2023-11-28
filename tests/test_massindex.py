import math
from random import random
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import MassIndex
from ta.trend import MassIndex as MassIndexReference


class MassIndexTest(TestCase):

    def test_against_random_reference(self):
        data = [sorted([random(), random()]) for _ in range(200)]

        smi = MassIndex(fillna=True)
        for i, (x,y) in enumerate(zip([smi.update(*x) for x in data],
                                      MassIndexReference(fillna=True, high=pd.Series(x[1] for x in data), low=pd.Series(x[0] for x in data)).mass_index())):
            self.assertAlmostEqual(x,y, places=4)

if __name__ == "__main__":
    main()
