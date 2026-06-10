# RollingSpreadLiquidityShiftDetector

## Summary

`RollingSpreadLiquidityShiftDetector` is RTTA's streaming implementation of: Causal adjacent-window quote spread/depth liquidity stress shift detector.

## Update API

```python
result = rtta.RollingSpreadLiquidityShiftDetector(window=20, threshold=1e-06).update(bid_price, bid_size, ask_price, ask_size)
```

The `update(...)` call consumes one observation using `bid_price`, `bid_size`, `ask_price`, `ask_size`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RollingSpreadLiquidityShiftDetector` compares two adjacent rolling windows: a reference window and a recent window. The C++ state moves expired recent samples into the reference window, maintains sufficient statistics, and emits the sign of the statistic difference when it exceeds the configured threshold.

## Recurrence

Let \(z_t = (bid_price_t, bid_size_t, ask_price_t, ask_size_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t=\frac{\max(ask_t-bid_t,0)}{\max(bid\_size_t+ask\_size_t,\epsilon)}
\]

\[
q_t=\operatorname{mean}(R^s_t)-\operatorname{mean}(B^s_t)
\]

\[
r_t =
\begin{cases}
1, & q_t > h \\
-1, & q_t < -h \\
0, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingSpreadLiquidityShiftDetector`.

## Reference

- [Background reference](https://arxiv.org/abs/1011.6402)
