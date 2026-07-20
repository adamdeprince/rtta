# BollingerBandwidth

## Summary

`BollingerBandwidth` is RTTA's streaming implementation of: Bollinger band width as (upper-lower)/middle for a rolling mean and standard-deviation envelope.

## Update API

```python
result = rtta.BollingerBandwidth().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BollingerBandwidth` normalizes the distance between the upper and lower
Bollinger bands by the middle band. Rising bandwidth marks volatility expansion;
falling bandwidth marks compression.

## Recurrence

Let \(z_t = value_t\), \(n\) the window, and \(k\) the standard-deviation multiplier (`num_std`, default 2).

\[
M_t = \operatorname{mean}_n(z_t), \qquad
S_t = \operatorname{stddev}_n(z_t)
\]

\[
BW_t = \frac{(M_t + k S_t) - (M_t - k S_t)}{M_t} = \frac{2 k S_t}{M_t}
\]

The return value is the current scalar indicator value.

## Composed Primitives

[`BollingerBands`](bollinger-bands.md), [`SMA`](sma.md), [`StdDev`](std-dev.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BollingerBandwidth`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/bollinger-bandwith)
