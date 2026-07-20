# RelativeVolatilityIndex

## Summary

`RelativeVolatilityIndex` is RTTA's streaming Donald Dorsey Relative Volatility
Index (RVI). It applies RSI-style smoothing to upward and downward standard
deviation of close, and also emits an EMA signal line of the RVI.

## Update API

```python
result = rtta.RelativeVolatilityIndex(
    std_window=10, smooth_window=14, fillna=True
).update(close)
# result.rvi, result.signal
```

With `fillna=False`, both fields are `NaN` until
`std_window + smooth_window` samples have been seen.

## Theory Of Operation

Dorsey's RVI looks like RSI but replaces price change with rolling standard
deviation of close, assigned to the up side when close rises and the down side
when close falls. The result measures the direction of volatility rather than
the direction of price: high RVI means volatility is occurring more on up moves;
low RVI means more on down moves. RTTA also smooths RVI with a same-length EMA
as a signal line.

## Recurrence

Let \(c_t\) be close, \(n_s\) be `std_window` (default \(10\)), and \(n_e\) be
`smooth_window` (default \(14\)).

\[
\sigma_t = \operatorname{StdDev}_{n_s}(c_t)
\]

Assign volatility to up/down sides (first bar contributes zeros):

\[
(u_t, d_t) =
\begin{cases}
(\sigma_t,\, 0), & c_t > c_{t-1} \\
(0,\, \sigma_t), & c_t < c_{t-1} \\
(0.5\,\sigma_t,\, 0.5\,\sigma_t), & c_t = c_{t-1}
\end{cases}
\]

\[
U_t = \operatorname{EMA}_{n_e}(u_t), \qquad
D_t = \operatorname{EMA}_{n_e}(d_t)
\]

\[
RVI_t = 100 \cdot \frac{U_t}{U_t + D_t}
\quad\text{(safe divide)}
\]

\[
\operatorname{signal}_t = \operatorname{EMA}_{n_e}(RVI_t)
\]

Nested StdDev and EMAs use `fillna=True`. Outer warm count is \(n_s + n_e\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class RelativeVolatilityIndex` with members `std_`, `up_`, `down_`, and
`signal_`. Result fields are `rvi` and `signal`. See also
[`Inertia`](inertia.md), which regresses RVI.

## Reference

- [Investopedia: Relative Volatility Index](https://www.investopedia.com/terms/r/relative-volatility-index-rvi.asp)
- [TradingView: Relative Volatility Index](https://www.tradingview.com/scripts/relativevolatilityindex/)
