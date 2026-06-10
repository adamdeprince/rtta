# OrderFlowImbalanceRegimeDetector

## Summary

`OrderFlowImbalanceRegimeDetector` is RTTA's streaming implementation of: EWMA order-flow imbalance regime detector with buy/sell pressure hysteresis.

## Update API

```python
result = rtta.OrderFlowImbalanceRegimeDetector().update(bid_price, bid_size, ask_price, ask_size)
```

The `update(...)` call consumes one observation using `bid_price`, `bid_size`, `ask_price`, `ask_size`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`OrderFlowImbalanceRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (bid_price_t, bid_size_t, ask_price_t, ask_size_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
e_t=
\mathbf{1}[bid_t\ge bid_{t-1}]bidSize_t
-\mathbf{1}[bid_t\le bid_{t-1}]bidSize_{t-1}
-\mathbf{1}[ask_t\le ask_{t-1}]askSize_t
+\mathbf{1}[ask_t\ge ask_{t-1}]askSize_{t-1}
\]

\[
n_t=\frac{e_t}{\max(bidSize_t+askSize_t,\epsilon)}, \qquad
q_t=\alpha n_t+(1-\alpha)q_{t-1}
\]

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

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class OrderFlowImbalanceRegimeDetector`.

## Reference

- [Background reference](https://arxiv.org/abs/1011.6402)
