# ComparativeRelativeStrength

## Summary

`ComparativeRelativeStrength` is RTTA's streaming price ratio of two series. It
returns \(price_a / price_b\) on every tick with no smoothing or window state.

## Update API

```python
value = rtta.ComparativeRelativeStrength().update(price_a, price_b)
```

There are no constructor parameters. Array `batch(...)` requires equal-length
`price_a` and `price_b` arrays.

## Theory Of Operation

Comparative (or relative) strength between two instruments is the ratio of their
prices. An rising ratio means series A is outperforming series B; a falling
ratio means A is underperforming. Traders often chart the ratio itself or apply
a moving average to the ratio (RTTA leaves that optional post-processing to the
caller).

Unlike RSI ("Relative Strength Index"), this is not an oscillator of gains and
losses on a single series — it is a direct comparative ratio.

## Recurrence

Let \(a_t\) and \(b_t\) be the two prices at time \(t\).

\[
CRS_t = \frac{a_t}{b_t}
\quad\text{(safe divide; \(0\) if \(b_t = 0\))}
\]

There is no internal state; each update is independent of prior bars.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ComparativeRelativeStrength`.

## Reference

- [ChartSchool: Price Relative / Relative Strength](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/price-relative-relative-strength)
- [Investopedia: Relative Strength](https://www.investopedia.com/terms/r/relativestrength.asp)
