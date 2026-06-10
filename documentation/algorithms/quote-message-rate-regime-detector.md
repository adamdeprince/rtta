# QuoteMessageRateRegimeDetector

## Summary

`QuoteMessageRateRegimeDetector` is RTTA's streaming implementation of: Relative EWMA quote-message-rate regime detector.

## Update API

```python
result = rtta.QuoteMessageRateRegimeDetector().update(quote_messages)
```

The `update(...)` call consumes one observation using `quote_messages`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`QuoteMessageRateRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = quote_messages_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
b_t=\alpha\max(x_t,0)+(1-\alpha)b_{t-1}
\]

\[
q_t=\frac{\max(x_t,0)}{\max(b_{t-1},\epsilon)}
\]

The C++ implementation evaluates the ratio against the prior EWMA baseline and then updates the baseline with the current observation.

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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class QuoteMessageRateRegimeDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Quote_stuffing)
