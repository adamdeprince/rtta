# CUSUM

## Summary

`CUSUM` is RTTA's streaming implementation of: Causal cumulative-sum event filter for detecting threshold-sized directional moves.

## Update API

```python
result = rtta.CUSUM(threshold=1.0, drift=0.0).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`CUSUM` implements the streaming form of Causal cumulative-sum event filter for detecting threshold-sized directional moves. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{CUSUM}(s_{t-1}, close_t; \theta)
\]

\[
y_t = G_{CUSUM}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CUSUM`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/CUSUM)
