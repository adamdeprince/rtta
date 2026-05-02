import math
from unittest import TestCase, main

from rtta.indicator import DoubleEMA, HullMovingAverage, T3MovingAverage, TriangularMovingAverage, TripleEMA, Trix, WeightedMovingAverage


class MovingAverageVariantTest(TestCase):
    def test_constant_series(self):
        for cls in (DoubleEMA, HullMovingAverage, T3MovingAverage, TripleEMA, TriangularMovingAverage):
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

    def test_hull_moving_average(self):
        hma = HullMovingAverage(window=4)
        values = [1, 2, 3, 4, 5]
        for value in values:
            result = hma.update(value)

        def wma(items):
            weights = range(1, len(items) + 1)
            return sum(weight * item for weight, item in zip(weights, items)) / sum(weights)

        transformed = []
        for index in range(len(values)):
            current = values[: index + 1]
            transformed.append(2 * wma(current[-2:]) - wma(current[-4:]))
        expected = wma(transformed[-2:])
        self.assertAlmostEqual(result, expected)

    def test_hull_moving_average_fillna_false(self):
        hma = HullMovingAverage(window=4, fillna=False)
        self.assertTrue(math.isnan(hma.update(1)))
        self.assertTrue(math.isnan(hma.update(2)))
        self.assertTrue(math.isnan(hma.update(3)))
        self.assertTrue(math.isnan(hma.update(4)))
        self.assertTrue(math.isfinite(hma.update(5)))


if __name__ == "__main__":
    main()
