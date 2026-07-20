# MovingAverageEnvelope

## Summary

`MovingAverageEnvelope` is RTTA's streaming implementation of: Percentage envelope bands above and below a simple moving average.

## Update API

```python
result = rtta.MovingAverageEnvelope().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MovingAverageEnvelope` places fixed percentage bands around a simple moving
average. Unlike Bollinger bands, the width does not depend on recent volatility;
`percent` is a fraction (for example `0.025` for 2.5% bands).

## Recurrence

Let \(z_t = value_t\), \(n\) the window, and \(p\) the envelope fraction.

\[
M_t = \operatorname{SMA}_n(z_t)
\]

\[
U_t = M_t(1+p), \qquad
L_t = M_t(1-p)
\]

`update(...)` returns a result struct with fields `middle`, `upper`, `lower`.

## Composed Primitives

[`SMA`](sma.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MovingAverageEnvelope`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-average-envelopes)
