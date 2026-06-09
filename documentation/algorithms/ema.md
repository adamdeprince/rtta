# EMA

## Summary

`EMA` computes an exponential moving average over one scalar stream. The object
stores only the latest EMA value, a sample counter, and the fixed smoothing
multiplier derived from `window`.

## Update API

```python
value = rtta.EMA(window=30.0, fillna=False).update(value)
```

The same scalar stream can be supplied through `update(...)`, `advance(...)`,
or `batch(...)`.

## Theory Of Operation

An exponential moving average gives the newest sample a fixed weight and carries
forward the prior average with the complementary weight. RTTA seeds the state
from the first input sample, so the first internal EMA value equals the first
input value.

## Recurrence

Let \(x_t\) be the new sample and \(n\) be `window`. RTTA clamps `window` to at
least 1 and uses:

\[
\alpha = \frac{2}{1+n}
\]

Initialization:

\[
E_0 = x_0
\]

For each later update:

\[
E_t = \alpha x_t + (1-\alpha)E_{t-1}
\]

With `fillna=False`, the state updates immediately, but returned values are
`NaN` until the internal counter reaches `window`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class EMA`.

## Reference

- [ChartSchool: Simple and Exponential Moving Averages](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
