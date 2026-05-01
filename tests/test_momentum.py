import math
from unittest import TestCase, main

from rtta.indicator import Momentum, RateOfChangePercentage, RateOfChangeRatio, RateOfChangeRatio100


class MomentumTest(TestCase):
    def test(self):
        momentum = Momentum(window=2, fillna=False)
        self.assertTrue(math.isnan(momentum.update(1)))
        self.assertTrue(math.isnan(momentum.update(2)))
        self.assertEqual(momentum.update(4), 3)

    def test_rate_of_change_variants(self):
        rocp = RateOfChangePercentage(window=1)
        self.assertEqual(rocp.update(2), 0)
        self.assertEqual(rocp.update(4), 1)

        rocr = RateOfChangeRatio(window=1)
        self.assertEqual(rocr.update(8), 0)
        self.assertEqual(rocr.update(16), 2)

        rocr100 = RateOfChangeRatio100(window=1)
        self.assertEqual(rocr100.update(16), 0)
        self.assertEqual(rocr100.update(32), 200)


if __name__ == "__main__":
    main()
