from unittest import TestCase, main

from rtta.indicator import Beta, Correlation, Variance


class StatisticsTest(TestCase):
    def test(self):
        correlation = Correlation(window=3)
        beta = Beta(window=3)
        variance = Variance(window=3)

        for x, y in [(2, 1), (4, 2), (6, 3)]:
            correlation_value = correlation.update(x, y)
            beta_value = beta.update(x, y)

        for value in [1, 2, 3]:
            variance_value = variance.update(value)

        self.assertAlmostEqual(correlation_value, 1)
        self.assertAlmostEqual(beta_value, 2)
        self.assertAlmostEqual(variance_value, 2 / 3)


if __name__ == "__main__":
    main()
