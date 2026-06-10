# MicrostructureNoiseRegimeDetector

## Summary

`MicrostructureNoiseRegimeDetector` is RTTA's streaming implementation of: EWMA trade-versus-mid noise detector normalized by quoted spread.

## Update API

```python
result = rtta.MicrostructureNoiseRegimeDetector().update(trade_price, bid_price, ask_price)
```

The `update(...)` call consumes one observation using `trade_price`, `bid_price`, `ask_price`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MicrostructureNoiseRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (trade_price_t, bid_price_t, ask_price_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
mid_t=\frac{bid_t+ask_t}{2}, \qquad
n_t=\frac{|trade_t-mid_t|}{\max(ask_t-bid_t,\epsilon)}
\]

\[
q_t=\alpha n_t+(1-\alpha)q_{t-1}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MicrostructureNoiseRegimeDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Market_microstructure)
