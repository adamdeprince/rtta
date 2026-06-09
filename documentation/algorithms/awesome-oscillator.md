# AwesomeOscillator

## Summary

`AwesomeOscillator` is RTTA's streaming implementation of: Difference between short and long median-price moving averages.

## Update API

```python
result = rtta.AwesomeOscillator().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`AwesomeOscillator` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class AwesomeOscillator`.

## Reference

- [Background reference](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
