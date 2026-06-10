# BidAskBounceRegimeDetector

## Summary

`BidAskBounceRegimeDetector` is RTTA's streaming implementation of: EWMA bid/ask side alternation detector for quote-bounce regimes.

## Update API

```python
result = rtta.BidAskBounceRegimeDetector().update(trade_price, bid_price, ask_price)
```

The `update(...)` call consumes one observation using `trade_price`, `bid_price`, `ask_price`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BidAskBounceRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (trade_price_t, bid_price_t, ask_price_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
side_t=\begin{cases}1, & trade_t\ge (bid_t+ask_t)/2\\ -1, & \text{otherwise}\end{cases}
\]

\[
b_t=\mathbf{1}[side_t\ne side_{t-1}], \qquad
q_t=\alpha b_t+(1-\alpha)q_{t-1}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BidAskBounceRegimeDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread)
