# ArnaudLegouxMovingAverage

## Summary

`ArnaudLegouxMovingAverage` is RTTA's streaming implementation of: Arnaud Legoux moving average with Gaussian weights controlled by offset and sigma.

## Update API

```python
result = rtta.ArnaudLegouxMovingAverage().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ArnaudLegouxMovingAverage` applies a fixed Gaussian window to the most recent
`window` samples. The peak of the Gaussian is placed at
`offset * (window - 1)` from the oldest sample, so larger `offset` values put
more weight on recent prices. `sigma` controls how peaked the kernel is.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call, \(n\) the window length, \(o\) the offset, and \(s=n/\sigma\).

\[
m = o(n-1), \qquad
w_i = \exp\!\left(-\frac{(i-m)^2}{2s^2}\right), \quad i=0,\ldots,n-1
\]

\[
y_t = \frac{\sum_{i=0}^{n-1} w_i z_{t-n+1+i}}{\sum_{i=0}^{n-1} w_i}
\]

Index \(i=0\) is the oldest sample in the window and \(i=n-1\) is the newest.
Until the window is full, only the available samples and their matching weights
are used when `fillna=True`.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ArnaudLegouxMovingAverage`.
Weights are precomputed in the constructor.

## Reference

- [Background reference](https://www.tradingview.com/support/solutions/43000594683-arnaud-legoux-moving-average/)
