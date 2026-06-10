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

`IntradayClockEchoSignal` learns time-of-day residual return patterns by slot. Each update removes an optional market return, updates the EWMA state for the current slot, forecasts a future slot path over the horizon, and calibrates forecast errors with a rolling quantile. The linked research note explains the same-clock effect and session-alignment assumptions.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
r_t=\log(close_t/close_{t-1}), \qquad
\epsilon_t=r_t-market\_return_t
\]

\[
E_{s,t}=(1-\alpha)E_{s,t-1}+\alpha\epsilon_t
\quad \text{for } s=slot_t
\]

\[
w_j=\exp(-0.10(j-1))\min\left(1,\frac{count_{slot_t+j}}{min\_slot\_samples}\right)
\]

\[
clock\_echo_t=
\frac{\sum_{j=1}^{h}w_jE_{slot_t+j,t}}{\sum_{j=1}^{h}w_j},
\qquad
\widehat{r}_{t+h}=h\cdot clock\_echo_t
\]

\[
\mathcal{E}_t=\{|r^{(h)}_i-\widehat{r}^{(h)}_i|:\ i+h\le t\}, \qquad
radius_t=\max(Q_{\tau}(\mathcal{E}_t), cost)
\]

\[
score_t=\frac{\widehat{r}_{t+h}}{radius_t+cost}
\]

`update(...)` returns a result struct with fields `slot`, `samples_for_slot`, `bar_return`, `residual_return`, `clock_echo`, `flow_confirm`, `volume_sync`, `prediction`, `radius`, `score`, `signal`, `target_fraction`, `max_trade_dollars`, `realized_error`, `ready`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class IntradayClockEchoSignal`.

## Reference

- [Detailed research note](../intraday_clock_echo_signal.md)
