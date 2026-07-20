# ZeroLagEMA

## Summary

`ZeroLagEMA` is RTTA's streaming implementation of: Zero-lag exponential moving average using de-lagged price into an EMA.

## Update API

```python
result = rtta.ZeroLagEMA().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ZeroLagEMA` reduces EMA lag by feeding a de-lagged series into a standard EMA.
The lag estimate is half the EMA period (integer), and the de-lagged input is
current price plus the difference between current and lagged price.

## Recurrence

Let \(z_t = value_t\), \(n\) the window, and \(L=\lfloor (n-1)/2 \rfloor\).

\[
\tilde{z}_t = 2z_t - z_{t-L}
\]

\[
y_t = \operatorname{EMA}_n(\tilde{z}_t)
\]

with the same EMA seeding rules as [`EMA`](ema.md). When \(L=0\),
\(\tilde{z}_t=z_t\).

The return value is the current scalar indicator value.

## Composed Primitives

[`EMA`](ema.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ZeroLagEMA`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average)
