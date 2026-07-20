# MovingAverageVariablePeriod

## Summary

`MovingAverageVariablePeriod` is RTTA's TA-Lib-style MAVP: a simple moving
average of `value` whose lookback length is supplied per bar and clamped to
`[min_period, max_period]`.

## Update API

```python
value = rtta.MovingAverageVariablePeriod(
    max_period=30, min_period=2, fillna=True
).update(value, period)
```

`period` is rounded to the nearest integer with `llround`, then clamped. With
`fillna=False`, output is `NaN` when fewer than the (clamped) period samples are
available.

## Theory Of Operation

Variable-period averages let another series — efficiency ratio, volatility,
cycle period, etc. — control the smoother's memory on each bar. RTTA keeps a
rolling buffer of length `max_period` and, each update, averages only the most
recent \(p\) samples where \(p\) is the clamped requested period. This matches
the common TA-Lib MAVP interpretation of a plain SMA with dynamic length.

## Recurrence

Let \(x_t\) be `value`, \(p^{\text{raw}}_t\) be `period`, \(p_{\min}\) be
`min_period`, and \(p_{\max}\) be `max_period`.

\[
p_t = \operatorname{clamp}\!\bigl(\operatorname{round}(p^{\text{raw}}_t),\, p_{\min},\, p_{\max}\bigr)
\]

Maintain a FIFO buffer of the last up to \(p_{\max}\) values. Let \(n_t\) be the
current buffer size and \(u_t = \min(n_t, p_t)\). Sum the most recent \(u_t\)
samples:

\[
MAVP_t = \frac{1}{u_t}\sum_{i=0}^{u_t-1} x_{t-i}
\]

If `fillna=False` and \(n_t < p_t\), return `NaN`. Defaults: \(p_{\max}=30\),
\(p_{\min}=2\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class MovingAverageVariablePeriod`. The buffer capacity is always `max_period`.

## Reference

- [TA-Lib: MAVP — Moving average with variable period](https://ta-lib.org/functions/MAVP/)
- [Investopedia: Moving Average](https://www.investopedia.com/terms/m/movingaverage.asp)
