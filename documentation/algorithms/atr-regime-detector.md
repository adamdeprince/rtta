# ATRRegimeDetector

## Summary

`ATRRegimeDetector` is RTTA's streaming implementation of: Stateful ATR regime detector with high/low hysteresis bands.

## Update API

```python
result = rtta.ATRRegimeDetector().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ATRRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
TR_t=\max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]

\[
q_t=ATR_t=\operatorname{WilderEMA}_n(TR_t)
\]

This recurrence composes the standard RTTA `ATR` update with the same two-sided hysteresis state used by `ThresholdRegimeDetector`.

\[
r_t =
\begin{cases}
1, & r_{t-1} \le 0 \text{ and } q_t \ge u_e \\
0, & r_{t-1} = 1 \text{ and } q_t \le u_x \\
-1, & r_{t-1} \ge 0 \text{ and } q_t \le \ell_e \\
0, & r_{t-1} = -1 \text{ and } q_t \ge \ell_x \\
r_{t-1}, & \text{otherwise}
\end{cases}
\]

The entry/exit constants satisfy \(\ell_e < \ell_x \le u_x < u_e\).

The return value is the current scalar indicator value.

## Composed Primitives

[`ATR`](atr.md), [`ThresholdRegimeDetector`](threshold-regime-detector.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ATRRegimeDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
