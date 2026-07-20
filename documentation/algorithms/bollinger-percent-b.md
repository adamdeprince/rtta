# BollingerPercentB

## Summary

`BollingerPercentB` is RTTA's streaming implementation of: Bollinger %B position of price inside a rolling mean and standard-deviation envelope.

## Update API

```python
result = rtta.BollingerPercentB().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BollingerPercentB` reports where the current price sits between the lower and
upper Bollinger bands. Values near 0 hug the lower band, near 1 hug the upper
band, and values outside \([0,1]\) are outside the envelope.

## Recurrence

Let \(z_t = value_t\), \(n\) the window, and \(k\) the standard-deviation multiplier (`num_std`, default 2).

\[
M_t = \operatorname{mean}_n(z_t), \qquad
S_t = \operatorname{stddev}_n(z_t)
\]

\[
U_t = M_t + k S_t, \qquad
L_t = M_t - k S_t, \qquad
\%B_t = \frac{z_t - L_t}{U_t - L_t}
\]

Variance uses the same population form as [`BollingerBands`](bollinger-bands.md)
(\(1/n\) mean of squares).

The return value is the current scalar indicator value.

## Composed Primitives

[`BollingerBands`](bollinger-bands.md), [`SMA`](sma.md), [`StdDev`](std-dev.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BollingerPercentB`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/bollinger-bandwidth-and-b)
