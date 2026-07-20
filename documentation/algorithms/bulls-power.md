# BullsPower

## Summary

`BullsPower` is RTTA's streaming implementation of Alexander Elder's bulls
power: the distance of the bar high above an EMA of close. It is the bull
component of `ElderRayIndex`, exposed as a single scalar for retail-style APIs.

## Update API

```python
value = rtta.BullsPower(window=13, fillna=True).update(high, close)
```

The `update(...)` call consumes `high` and `close`. `advance(...)` uses the same
inputs without returning a Python value. Scalar `batch(high, close)` returns a
NumPy array.

## Theory Of Operation

Elder models "bulls" as how far buyers managed to push the high relative to a
consensus value of close (the EMA). Large positive bulls power means the high
stands well above the smoothed close; values near zero mean the high is pinned
to the EMA. Combined with `BearsPower` (or `ElderRayIndex`), it forms the classic
Elder-ray picture of bull and bear pressure.

## Recurrence

Let \(c_t\) be close, \(h_t\) be high, and \(n\) be `window`. Let
\(\operatorname{EMA}_n\) be RTTA's exponential moving average of close.

\[
E_t = \operatorname{EMA}_n(c_t)
\]

\[
\operatorname{Bulls}_t = h_t - E_t
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BullsPower`.
On the same EMA seed path it matches `ElderRayIndex.bull_power`.

## Reference

- [Investopedia: Elder-Ray Index](https://www.investopedia.com/articles/trading/03/022603.asp)
- [ChartSchool: Elder-Ray Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/elder-ray-index)
