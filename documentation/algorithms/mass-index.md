# MassIndex

## Summary

`MassIndex` is RTTA's streaming implementation of: Range-expansion reversal indicator.

## Update API

```python
result = rtta.MassIndex().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MassIndex` studies range expansion rather than direction. It double-smooths the high-low range with `EMA`, forms their ratio, and sums that ratio over a window so persistent range bulges become visible as a reversal-risk signal.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
R_t=high_t-low_t, \qquad
E_t=\operatorname{EMA}_n(R_t), \qquad
D_t=\operatorname{EMA}_n(E_t)
\]

\[
y_t=\sum_{i\in W_t}\frac{E_i}{D_i}
\]

The return value is the current scalar indicator value.

## Composed Primitives

[`EMA`](ema.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MassIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/mass-index)
