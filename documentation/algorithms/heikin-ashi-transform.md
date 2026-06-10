# HeikinAshiTransform

## Summary

`HeikinAshiTransform` is RTTA's streaming implementation of: Incremental Heikin-Ashi OHLC transform for smoothing candles.

## Update API

```python
result = rtta.HeikinAshiTransform().update(open, high, low, close)
```

The `update(...)` call consumes one observation using `open`, `high`, `low`, `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`HeikinAshiTransform` implements the streaming form of Incremental Heikin-Ashi OHLC transform for smoothing candles. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
HAclose_t=\frac{open_t+high_t+low_t+close_t}{4}
\]

\[
HAopen_t=\frac{HAopen_{t-1}+HAclose_{t-1}}{2}
\]

\[
HAhigh_t=\max(high_t,HAopen_t,HAclose_t), \qquad
HAlow_t=\min(low_t,HAopen_t,HAclose_t)
\]

`update(...)` returns a result struct with fields `open`, `high`, `low`, `close`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class HeikinAshiTransform`.

## Reference

- [Background reference](https://www.mql5.com/en/articles/19260)
