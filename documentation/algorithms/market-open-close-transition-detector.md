# MarketOpenCloseTransitionDetector

## Summary

`MarketOpenCloseTransitionDetector` is RTTA's streaming implementation of: Session-progress transition detector for market-open and market-close bands.

## Update API

```python
result = rtta.MarketOpenCloseTransitionDetector().update(session_progress)
```

The `update(...)` call consumes one observation using `session_progress`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MarketOpenCloseTransitionDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = session_progress_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
p_t=\operatorname{clip}(session\_progress_t,0,1)
\]

\[
r_t =
\begin{cases}
1, & r_{t-1}=0 \text{ and } p_t\le open_e\\
0, & r_{t-1}=1 \text{ and } p_t\ge open_x\\
-1, & r_{t-1}=0 \text{ and } p_t\ge close_e\\
0, & r_{t-1}=-1 \text{ and } p_t\le close_x\\
r_{t-1}, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MarketOpenCloseTransitionDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Trading_day)
