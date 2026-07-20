# BearsPower

## Summary

`BearsPower` is RTTA's streaming implementation of Alexander Elder's bears
power: the distance of the bar low below an EMA of close. It is the bear
component of `ElderRayIndex`, exposed as a single scalar for retail-style APIs.

## Update API

```python
value = rtta.BearsPower(window=13, fillna=True).update(low, close)
```

The `update(...)` call consumes `low` and `close`. `advance(...)` uses the same
inputs without returning a Python value. Scalar `batch(low, close)` returns a
NumPy array.

## Theory Of Operation

Elder models "bears" as how far sellers managed to push the low relative to a
consensus value of close (the EMA). Large negative bears power means the low is
well below the smoothed close; values near zero mean the low is hugging the EMA.
Together with `BullsPower`, it is the two-sided Elder-ray pressure view.

## Recurrence

Let \(c_t\) be close, \(\ell_t\) be low, and \(n\) be `window`. Let
\(\operatorname{EMA}_n\) be RTTA's exponential moving average of close.

\[
E_t = \operatorname{EMA}_n(c_t)
\]

\[
\operatorname{Bears}_t = \ell_t - E_t
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BearsPower`.
On the same EMA seed path it matches `ElderRayIndex.bear_power`.

## Reference

- [Investopedia: Elder-Ray Index](https://www.investopedia.com/articles/trading/03/022603.asp)
- [ChartSchool: Elder-Ray Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/elder-ray-index)
