# IntradayClockEchoSignal

## Summary

`IntradayClockEchoSignal` is RTTA's streaming implementation of: Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists.

## Update API

```python
result = rtta.IntradayClockEchoSignal(fillna=True).update(open, high, low, close, volume)
```

The `update(...)` call consumes one observation using `open`, `high`, `low`, `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`IntradayClockEchoSignal` implements the streaming form of Same-clock intraday return-periodicity signal trained from prior aggregate-bar day lists. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{IntradayClockEchoSignal}(s_{t-1}, (open_t, high_t, low_t, close_t, volume_t); \theta)
\]

\[
y_t = G_{IntradayClockEchoSignal}(s_t)
\]

`update(...)` returns a result struct with fields `slot`, `samples_for_slot`, `bar_return`, `residual_return`, `clock_echo`, `flow_confirm`, `volume_sync`, `prediction`, `radius`, `score`, `signal`, `target_fraction`, `max_trade_dollars`, `realized_error`, `ready`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class IntradayClockEchoSignal`.

## Reference

- [intraday_clock_echo_signal](../intraday_clock_echo_signal.md)
