# ChaikinVolatility

## Summary

`ChaikinVolatility` is RTTA's streaming implementation of: Percent rate-of-change of an EMA of the high-low range.

## Update API

```python
result = rtta.ChaikinVolatility().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ChaikinVolatility` first smooths the bar range with an EMA, then reports the
percent change of that smoothed range versus its value `roc_window` samples
ago. Rising values mark expanding range; falling values mark contraction.

## Recurrence

Let \(h_t,l_t\) be high and low, \(n\) the EMA window, and \(k\) the ROC lookback.

\[
R_t = h_t - l_t, \qquad
E_t = \operatorname{EMA}_n(R_t)
\]

\[
CV_t = 100 \cdot \frac{E_t - E_{t-k}}{E_{t-k}}
\]

The return value is the current scalar indicator value.

## Composed Primitives

[`EMA`](ema.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ChaikinVolatility`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-volatility)
