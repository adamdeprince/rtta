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
\Delta_t=close_t-close_{t-1}
\]

\[
S^+_t=\max(0,S^+_{t-1}+\Delta_t-\kappa), \qquad
S^-_t=\min(0,S^-_{t-1}+\Delta_t+\kappa)
\]

\[
y_t=\begin{cases}1,&S^+_t>h\\-1,&S^-_t<-h\\0,&\text{otherwise}\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CUSUM`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/CUSUM)
