# SchaffTrendCycle

## Summary

`SchaffTrendCycle` is RTTA's streaming implementation of: MACD/stochastic cycle oscillator.

## Update API

```python
result = rtta.SchaffTrendCycle().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SchaffTrendCycle` normalizes recent directional movement into an oscillator. The implementation keeps only causal rolling or smoothed state and maps that state into the current oscillator value.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SchaffTrendCycle`.

## Reference

- [Background reference](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
