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

`MicrostructureNoiseRegimeDetector` converts each observation into a streaming score and then applies threshold or hysteresis logic. The state is deliberately sticky where the C++ class models regimes, so small reversals do not immediately flip the output.

## Recurrence

Let \(z_t = (trade_price_t, bid_price_t, ask_price_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F(s_{t-1}, z_t)
\]

\[
r_t =
\begin{cases}
1, & score(s_t) \ge u \\
-1, & score(s_t) \le l \\
r_{t-1}, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MicrostructureNoiseRegimeDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Market_microstructure)
