# MatchedFlowConformalSignal

## Summary

`MatchedFlowConformalSignal` is RTTA's streaming implementation of: Intraday OHLCV matched-flow signal with conformal-style rolling error bands and target sizing diagnostics.

## Update API

```python
result = rtta.MatchedFlowConformalSignal().update(open, high, low, close, volume)
```

The `update(...)` call consumes one observation using `open`, `high`, `low`, `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MatchedFlowConformalSignal` is a composite intraday research signal. It forms a horizon return forecast from multi-scale momentum, signed dollar-flow participation, VWAP displacement, and abnormal activity; then it scales that forecast by a rolling empirical error quantile. The longer research note linked below discusses the paper lineage and the conformal-style calibration caveats.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
r^{(k)}_t=\log(close_t/close_{t-k}), \qquad
m_t=0.20r^{(3)}_t+0.35r^{(6)}_t+0.45r^{(12)}_t
\]

\[
DV_t=close_t\max(volume_t,0), \qquad
relvol_t=\frac{DV_t}{normal\_dollar\_volume_t}
\]

\[
a_t=\operatorname{sgn}(r^{(1)}_t)
\frac{close_t\,\max(volume_t,0)}{\operatorname{scale}_t}, \qquad
p_t=\operatorname{sgn}(r^{(1)}_t)
\frac{close_t\,\max(volume_t,0)}{normal\_dollar\_volume_t}
\]

Here \(\operatorname{scale}_t\) is market capitalization when supplied and the
normal dollar-volume baseline otherwise.

\[
flow_t=\tanh\left(\frac{\sum_{i\in W^{12}_t}a_i}{\alpha_{norm}}
+0.5\frac{\sum_{i\in W^{6}_t}p_i}{6}\right), \qquad
vwap\_gap_t=\frac{close_t}{VWAP_t}-1
\]

\[
\widehat{r}_{t+h}=
\frac{0.35m_t+0.001flow_t+0.05vwap\_gap_t
+0.0005\tanh((relvol_t-1)/2)}
{1+25\max(high_t-low_t,0)/close_t}
\]

\[
\mathcal{E}_t=\{|r^{(h)}_i-\widehat{r}^{(h)}_i|:\ i+h\le t\}, \qquad
radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_{t+h}}{radius_t+cost}
\]

`update(...)` returns a result struct with fields `prediction`, `radius`, `score`, `signal`, `target_fraction`, `alpha_flow`, `participation`, `flow_score`, `momentum`, `volatility`, `vwap_gap`, `rel_dollar_volume`, `max_trade_dollars`, `realized_error`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MatchedFlowConformalSignal`.

## Reference

- [Detailed research note](../matched_flow_conformal_signal.md)
