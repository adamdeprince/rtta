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

`MatchedFlowConformalSignal` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
W_t = \operatorname{push}(W_{t-1}, z_t, n)
\]

\[
y_t = G(W_t)
\]

`update(...)` returns a result struct with fields `prediction`, `radius`, `score`, `signal`, `target_fraction`, `alpha_flow`, `participation`, `flow_score`, `momentum`, `volatility`, `vwap_gap`, `rel_dollar_volume`, `max_trade_dollars`, `realized_error`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MatchedFlowConformalSignal`.

## Reference

- [matched_flow_conformal_signal](../matched_flow_conformal_signal.md)
