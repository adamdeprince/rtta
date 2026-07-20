# SmoothedMovingAverage

## Summary

`SmoothedMovingAverage` is RTTA's streaming implementation of: Wilder/SMMA/RMA smoothed moving average seeded by an initial SMA window.

## Update API

```python
result = rtta.SmoothedMovingAverage().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SmoothedMovingAverage` is the classic Wilder smoother used by RSI and ATR,
also known as SMMA or TradingView's RMA. The first full window is a simple
average; subsequent values use Wilder's recursive form with effective alpha
\(1/n\).

## Recurrence

Let \(z_t = value_t\) and \(n\) the window length.

For the first \(n\) samples, seed with a simple average:

\[
S_n = \frac{1}{n}\sum_{i=0}^{n-1} z_{i+1}
\]

Thereafter:

\[
S_t = \frac{S_{t-1}(n-1) + z_t}{n}
\]

which is equivalent to \(S_t = \alpha z_t + (1-\alpha)S_{t-1}\) with
\(\alpha = 1/n\).

When `fillna=True`, partial simple averages are returned before the first full
window; when `fillna=False`, those samples are NaN.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SmoothedMovingAverage`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
