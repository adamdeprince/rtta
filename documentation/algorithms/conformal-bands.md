# ConformalBands

## Summary

`ConformalBands` is RTTA's streaming split-conformal-style prediction band: an
SMA center with a rolling quantile of absolute one-step residuals as radius.
It is a lightweight band primitive, not a full adaptive conformal inference
stack.

## Update API

```python
result = rtta.ConformalBands(window=20, alpha=0.1, fillna=True).update(value)
# result.middle, result.upper, result.lower, result.radius
```

`alpha` is the target non-coverage rate; the residual quantile level is
\(1-\alpha\). Multi-output `batch(...)` returns arrays for `middle`, `upper`,
`lower`, and `radius`.

## Theory Of Operation

Conformal prediction forms a prediction set by calibrating nonconformity scores
on recent data. For a simple one-step forecast, a common score is absolute
residual \(\lvert y_t - \hat y_{t\mid t-1}\rvert\). RTTA uses the previous SMA
value as \(\hat y\), stores absolute residuals in a rolling window, and takes a
quantile at level \(1-\alpha\) as the radius around the current SMA center.
This is intentionally simpler than adaptive conformal inference under
distribution shift; it is a causal rolling calibration band.

## Recurrence

Let \(x_t\) be the input, \(n\) be `window`, and \(q = 1-\alpha\).

1. If a previous prediction \(\hat x_{t-1}\) exists, push residual
   \(s_t = \lvert x_t - \hat x_{t-1}\rvert\) into a rolling quantile store of
   capacity \(n\).
2. Update the center:

\[
m_t = \operatorname{SMA}_n(x_t)
\]

3. Set \(\hat x_t \leftarrow m_t\) for the next residual.
4. Let \(R_t\) be the empirical quantile of stored absolute residuals at level
   \(q\) (RTTA `RollingQuantile`). Then

\[
\begin{aligned}
\operatorname{middle}_t &= m_t \\
\operatorname{radius}_t &= R_t \\
\operatorname{upper}_t &= m_t + R_t \\
\operatorname{lower}_t &= m_t - R_t
\end{aligned}
\]

With `fillna=False`, outputs are `NaN` until roughly \(n\) samples have been
seen. Before residuals exist, the radius is treated as \(0\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ConformalBands` using `SMA` and `RollingQuantile`. For a richer
OHLCV composite that also uses residual quantiles for sizing, see
`MatchedFlowConformalSignal`.

## Reference

- [Xu and Xie, "Sequential Predictive Conformal Inference for Time Series,"
  arXiv:2212.03463](https://arxiv.org/abs/2212.03463)
- [Gibbs and Candès, "Adaptive Conformal Inference Under Distribution Shift,"
  arXiv:2106.00170](https://arxiv.org/abs/2106.00170)
