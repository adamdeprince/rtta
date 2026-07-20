# Inertia

## Summary

`Inertia` is RTTA's streaming implementation of Donald Dorsey's inertia
indicator: a linear regression of the Relative Volatility Index (RVI). It
smooths volatility-direction into a slower trend-of-volatility feature.

## Update API

```python
value = rtta.Inertia(
    std_window=10, smooth_window=14, reg_window=20, fillna=True
).update(close)
```

The `update(...)` call consumes one close. `advance(...)` updates state without
returning a Python value. Scalar `batch(...)` returns a NumPy array.

## Theory Of Operation

Dorsey's Relative Volatility Index is an RSI-style oscillator applied to the
rolling standard deviation of close (up-vol vs down-vol). Inertia then applies a
rolling linear regression to that RVI series so the output tracks the local
trend of relative volatility rather than the raw RVI tick. High inertia means
volatility has been persistently biased to the upside of the RVI construction;
low inertia means the opposite.

## Recurrence

Let \(c_t\) be close. First form Dorsey RVI as in `RelativeVolatilityIndex`
with parameters `std_window` and `smooth_window`:

\[
\sigma_t = \operatorname{StdDev}_{n_\sigma}(c_t)
\]

\[
u_t =
\begin{cases}
\sigma_t & c_t > c_{t-1} \\
\tfrac12\sigma_t & c_t = c_{t-1} \\
0 & c_t < c_{t-1}
\end{cases}
\qquad
d_t =
\begin{cases}
\sigma_t & c_t < c_{t-1} \\
\tfrac12\sigma_t & c_t = c_{t-1} \\
0 & c_t > c_{t-1}
\end{cases}
\]

\[
U_t = \operatorname{EMA}_{n_s}(u_t),\quad
D_t = \operatorname{EMA}_{n_s}(d_t),\quad
\operatorname{RVI}_t = 100\cdot\frac{U_t}{U_t+D_t}
\]

Then inertia is the fitted linear-regression value of RVI over `reg_window`:

\[
I_t = \operatorname{LinReg}_{n_r}(\operatorname{RVI})_t.
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Inertia`,
composing `RelativeVolatilityIndex` with `LinearRegressionCore`.

## Reference

- [FM Labs: Relative Volatility Index](https://www.fmlabs.com/reference/default.htm?url=RVI.htm)
- [Investopedia: Relative Volatility Index](https://www.investopedia.com/terms/r/relativevolatilityindex.asp)
