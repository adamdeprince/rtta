# ChandeMomentumOscillator

## Summary

`ChandeMomentumOscillator` is RTTA's streaming implementation of: Momentum oscillator using sums of recent gains and losses.

## Update API

```python
result = rtta.ChandeMomentumOscillator().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ChandeMomentumOscillator` normalizes recent directional movement into an oscillator. The implementation keeps only causal rolling or smoothed state and maps that state into the current oscillator value.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}
{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ChandeMomentumOscillator`.

## Reference

- [Background reference](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo)
