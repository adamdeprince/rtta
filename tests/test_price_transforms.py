from unittest import TestCase, main

from rtta.indicator import AveragePrice, BalanceOfPower, MedianPrice, TypicalPrice, WeightedClosePrice


class PriceTransformTest(TestCase):
    def test(self):
        self.assertEqual(AveragePrice().update(1, 2, 0, 3), 1.5)
        self.assertEqual(MedianPrice().update(4, 2), 3)
        self.assertEqual(TypicalPrice().update(3, 5, 1), 3)
        self.assertEqual(WeightedClosePrice().update(3, 5, 1), 3)
        self.assertEqual(BalanceOfPower().update(1, 3, 1, 2), 0.5)


if __name__ == "__main__":
    main()
