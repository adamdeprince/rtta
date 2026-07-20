# EfficiencyRatio

## Summary

`EfficiencyRatio` is RTTA's streaming implementation of: Kaufman efficiency ratio of net directional move to path length over a rolling window.

## Update API

```python
result = rtta.EfficiencyRatio().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`EfficiencyRatio` is the direction/noise ratio that drives
[`Kama`](kama.md). Values near 1 mean a clean directional move; values near 0
mean a noisy path with little net displacement.

## Recurrence

Let \(c_t = close_t\) and \(n\) the window.

\[
ER_t = \frac{|c_t - c_{t-n}|}{\sum_{i=0}^{n-1} |c_{t-i} - c_{t-i-1}|}
\]

The denominator is a rolling sum of absolute one-step changes. The numerator
uses the close from \(n\) samples earlier, matching the efficiency term inside
KAMA.

The return value is the current scalar indicator value.

## Composed Primitives

[`Kama`](kama.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class EfficiencyRatio`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama)
