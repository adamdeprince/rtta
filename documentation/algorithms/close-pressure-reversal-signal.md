# ClosePressureReversalSignal

## Summary

`ClosePressureReversalSignal` is RTTA's streaming implementation of: End-of-day cross-sectional reversal signal using rest-of-day return, volume/transaction pressure, VWAP location, and rolling conformal-style error bands.

## Update API

```python
result = rtta.ClosePressureReversalSignal().update(open, high, low, close, volume)
```

The `update(...)` call consumes one observation using `open`, `high`, `low`, `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ClosePressureReversalSignal` converts the end-of-day reversal idea into a causal bar stream. It freezes the session return at a configured cutoff, normalizes loser/winner pressure by realized intraday volatility, adjusts for volume, transactions, and VWAP location, and only emits entries during the late-session window. The longer research note linked below gives the empirical motivation and parameter interpretation.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
ROD_t=\log(close_t)-\log(anchor), \qquad
F_t=ROD_{t_c}
\]

\[
DV_t=close_t\max(volume_t,0), \qquad
vwap\_gap_t=\frac{close_t}{VWAP_t}-1
\]

\[
\sigma_{intra,t}=\sqrt{N_{t_c}\operatorname{Var}(r^{(1)}_1,\ldots,r^{(1)}_{t_c})},
\qquad
L_t=\frac{\max(0,-F_t)}{\sigma_{intra,t}}, \quad
W_t=\frac{\max(0,F_t)}{\sigma_{intra,t}}
\]

\[
M^V_t=1+0.20\,\operatorname{clip}(\log(DV_t/NDV_t),-2,4), \qquad
M^X_t=1+0.10\,\operatorname{clip}(\log(X_t/NX_t),-2,4)
\]

\[
P^{long}_t=L_tM^V_tM^X_t
\left(1+0.50\,\operatorname{clip}\left(\frac{-vwap\_gap_t}{\sigma_{intra,t}},0,3\right)\right)
\]

\[
\widehat{r}_t=slope\cdot \max(0,-F_t)\,
\operatorname{clip}(P^{long}_t/2,0,2)
\]

\[
\mathcal{E}_t=\{|r^{entry\to exit}_i-\widehat{r}_i|:\ i \text{ has matured}\}, \qquad
radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_t}{radius_t+cost}
\]

`update(...)` returns a result struct with fields `bar_number`, `rod_return`, `frozen_rod_return`, `loser_z`, `winner_z`, `range_z`, `volume_shock`, `transaction_shock`, `vwap_gap`, `pressure_score`, `prediction`, `radius`, `score`, `signal`, `target_fraction`, `max_trade_dollars`, `realized_error`, `entry_window`, `exit_window`, `frozen`, `news_guard`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ClosePressureReversalSignal`.

## Reference

- [Detailed research note](../close_pressure_reversal_signal.md)
