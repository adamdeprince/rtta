# StdDev

## Summary

`StdDev` is RTTA's streaming implementation of: Rolling standard deviation.

## Update API

```python
result = rtta.StdDev(window=5).update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`StdDev` implements the streaming form of Rolling standard deviation. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\mu_t=\frac{1}{|W_t|}\sum_{i\in W_t}x_i
\]

\[
\sigma_t^2=\frac{1}{|W_t|}\sum_{i\in W_t}(x_i-\mu_t)^2
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class StdDev`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
