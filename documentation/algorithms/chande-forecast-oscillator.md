# ChandeForecastOscillator

## Summary

`ChandeForecastOscillator` is RTTA's streaming implementation of Tushar Chande's
forecast oscillator: the percent distance of the current close from a one-step
linear-regression time-series forecast (TSF).

## Update API

```python
value = rtta.ChandeForecastOscillator(window=14, fillna=True).update(close)
```

The `update(...)` call consumes one close. `advance(...)` uses the same input
without returning a Python value. Scalar `batch(...)` returns a NumPy array.

## Theory Of Operation

A rolling least-squares line over the last \(n\) closes produces a one-bar-ahead
forecast (the same quantity exposed by `TimeSeriesForecast`). The forecast
oscillator asks how far the live close sits above or below that forecast, as a
percent of price. Positive values mean price is above the fitted extrapolation;
negative values mean it is below.

## Recurrence

Let \(x_t\) be close and \(n\) be `window`. Over the rolling window of the last
\(n\) samples with abscissae \(0,\ldots,n-1\), fit

\[
\hat\beta_t = (X^\top X)^{-1} X^\top y_t
\]

and form the one-step forecast

\[
\operatorname{TSF}_t = [1,\, n]\,\hat\beta_t
\]

(as in RTTA's `LinearRegressionCore`, where `tsf` is the intercept plus slope
times \(n\)). The oscillator is

\[
\operatorname{CFO}_t = 100\cdot\frac{x_t - \operatorname{TSF}_t}{x_t}.
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ChandeForecastOscillator` using `LinearRegressionCore` and its `tsf`
field.

## Reference

- [Investopedia: Forecast Oscillator](https://www.investopedia.com/terms/f/forecasto.asp)
- [ChartSchool: Time Series Forecast / Linear Regression](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
