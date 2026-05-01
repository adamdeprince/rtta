import math
from unittest import TestCase, main

from rtta.indicator import DoubleEMA, T3MovingAverage, TriangularMovingAverage, TripleEMA, Trix, WeightedMovingAverage


class MovingAverageVariantTest(TestCase):
    def test_constant_series(self):
        for cls in (DoubleEMA, T3MovingAverage, TripleEMA, TriangularMovingAverage):
            indicator = cls()
            for _ in range(8):
                value = indicator.update(5)
            self.assertAlmostEqual(value, 5)

    def test_weighted_moving_average(self):
        wma = WeightedMovingAverage(window=3)
        for value in [1, 2, 3]:
            result = wma.update(value)
        self.assertAlmostEqual(result, 14 / 6)

    def test_trix(self):
        trix = Trix(window=3)
        self.assertEqual(trix.update(1), 0)
        self.assertTrue(math.isfinite(trix.update(2)))


if __name__ == "__main__":
    main()
