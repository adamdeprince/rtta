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

`ClosePressureReversalSignal` combines price, volume, and/or quote information into a streaming microstructure or participation measure. The update path advances only from the latest tick and prior state.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

`update(...)` returns a result struct with fields `bar_number`, `rod_return`, `frozen_rod_return`, `loser_z`, `winner_z`, `range_z`, `volume_shock`, `transaction_shock`, `vwap_gap`, `pressure_score`, `prediction`, `radius`, `score`, `signal`, `target_fraction`, `max_trade_dollars`, `realized_error`, `entry_window`, `exit_window`, `frozen`, `news_guard`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ClosePressureReversalSignal`.

## Reference

- [close_pressure_reversal_signal](../close_pressure_reversal_signal.md)
