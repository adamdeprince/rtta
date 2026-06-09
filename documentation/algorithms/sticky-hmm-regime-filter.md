# StickyHMMRegimeFilter

## Summary

`StickyHMMRegimeFilter` is RTTA's streaming implementation of: Online Gaussian HMM regime filter with high self-transition persistence.

## Update API

```python
result = rtta.StickyHMMRegimeFilter().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`StickyHMMRegimeFilter` converts each observation into a streaming score and then applies threshold or hysteresis logic. The state is deliberately sticky where the C++ class models regimes, so small reversals do not immediately flip the output.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\tilde{\pi}_t = A^\top \pi_{t-1}
\]

\[
\pi_t(i)=
\frac{\tilde{\pi}_t(i)\,p(z_t\mid i)}
{\sum_j \tilde{\pi}_t(j)\,p(z_t\mid j)}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class StickyHMMRegimeFilter`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Hidden_Markov_model)
