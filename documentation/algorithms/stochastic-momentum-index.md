# StochasticMomentumIndex

## Summary

`StochasticMomentumIndex` is RTTA's streaming implementation of: Double-smoothed stochastic momentum index with signal line.

## Update API

```python
result = rtta.StochasticMomentumIndex().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`StochasticMomentumIndex` measures where close sits relative to the midpoint of
the recent high-low range, then double-smooths both the distance and the range
before normalizing. A signal EMA is also produced.

## Recurrence

Let \(c_t,h_t,l_t\) be close/high/low, \(n\) the range window, and
\(s_1,s_2,s\) the smoothers.

\[
HH_t=\max_{0\le i<n}h_{t-i},\quad
LL_t=\min_{0\le i<n}l_{t-i},\quad
M_t=\frac{HH_t+LL_t}{2}
\]

\[
D_t=c_t-M_t,\quad
R_t=HH_t-LL_t
\]

\[
SM_t=\operatorname{EMA}_{s_2}(\operatorname{EMA}_{s_1}(D_t)),\quad
SR_t=\operatorname{EMA}_{s_2}(\operatorname{EMA}_{s_1}(R_t))
\]

\[
SMI_t=100\frac{SM_t}{0.5\,SR_t},\quad
signal_t=\operatorname{EMA}_s(SMI_t)
\]

`update(...)` returns a result struct with fields `smi`, `signal`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class StochasticMomentumIndex`.

## Reference

- [Background reference](https://www.investopedia.com/terms/s/stochmomentum.asp)
