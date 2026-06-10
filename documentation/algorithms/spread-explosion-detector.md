# SpreadExplosionDetector

## Summary

`SpreadExplosionDetector` is RTTA's streaming implementation of: EWMA relative quoted-spread explosion detector.

## Update API

```python
result = rtta.SpreadExplosionDetector().update(bid_price, ask_price)
```

The `update(...)` call consumes one observation using `bid_price`, `ask_price`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SpreadExplosionDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (bid_price_t, ask_price_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t=\max(ask_t-bid_t,0), \qquad
q_t=\frac{s_t}{\max(B_{t-1},\epsilon)}
\]

\[
B_t=\alpha s_t+(1-\alpha)B_{t-1}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SpreadExplosionDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread)
