from unittest import TestCase, main

from rtta.indicator import (
    LinearRegression,
    LinearRegressionAngle,
    LinearRegressionIntercept,
    LinearRegressionSlope,
    TimeSeriesForecast,
)


class LinearRegressionTest(TestCase):
    def test(self):
        linear = LinearRegression(window=3)
        slope = LinearRegressionSlope(window=3)
        intercept = LinearRegressionIntercept(window=3)
        angle = LinearRegressionAngle(window=3)
        forecast = TimeSeriesForecast(window=3)

        for value in [1, 2, 3]:
            linear_value = linear.update(value)
            slope_value = slope.update(value)
            intercept_value = intercept.update(value)
            angle_value = angle.update(value)
            forecast_value = forecast.update(value)

        self.assertAlmostEqual(linear_value, 3)
        self.assertAlmostEqual(slope_value, 1)
        self.assertAlmostEqual(intercept_value, 1)
        self.assertAlmostEqual(angle_value, 45)
        self.assertAlmostEqual(forecast_value, 4)


if __name__ == "__main__":
    main()
