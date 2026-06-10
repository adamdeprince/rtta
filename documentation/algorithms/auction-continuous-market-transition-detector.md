# AuctionContinuousMarketTransitionDetector

## Summary

`AuctionContinuousMarketTransitionDetector` is RTTA's streaming implementation of: Hysteresis detector for auction-versus-continuous market phase signals.

## Update API

```python
result = rtta.AuctionContinuousMarketTransitionDetector().update(auction_signal)
```

The `update(...)` call consumes one observation using `auction_signal`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`AuctionContinuousMarketTransitionDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = auction_signal_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
q_t=auction\_signal_t
\]

\[
r_t =
\begin{cases}
1, & r_{t-1} = 0 \text{ and } q_t \ge e \\
0, & r_{t-1} = 1 \text{ and } q_t \le x \\
r_{t-1}, & \text{otherwise}
\end{cases}, \qquad x < e
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class AuctionContinuousMarketTransitionDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Call_market)
