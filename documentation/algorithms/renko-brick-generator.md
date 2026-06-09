# RenkoBrickGenerator

## Summary

`RenkoBrickGenerator` is RTTA's streaming implementation of: Event-driven Renko price transform that emits signed brick counts and current brick state from close updates.

## Update API

```python
result = rtta.RenkoBrickGenerator().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RenkoBrickGenerator` implements the streaming form of Event-driven Renko price transform that emits signed brick counts and current brick state from close updates. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{RenkoBrickGenerator}(s_{t-1}, close_t; \theta)
\]

\[
y_t = G_{RenkoBrickGenerator}(s_t)
\]

`update(...)` returns a result struct with fields `brick_open`, `brick_close`, `direction`, `bricks`, `reversal`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RenkoBrickGenerator`.

## Reference

- [Background reference](https://www.tradingview.com/support/solutions/43000502284-understanding-renko-charts/)
