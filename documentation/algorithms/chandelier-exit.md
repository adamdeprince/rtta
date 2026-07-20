# ChandelierExit

## Summary

`ChandelierExit` is RTTA's streaming Chandelier Exit: ATR-based trailing long
and short stop levels hung from the rolling highest high and lowest low.

## Update API

```python
result = rtta.ChandelierExit(window=22, multiplier=3.0, fillna=True).update(
    close, high, low
)
# result.long_exit, result.short_exit
```

With `fillna=False`, both fields are `NaN` until `window` samples have been
seen and ATR is available.

## Theory Of Operation

Chuck LeBeau's Chandelier Exit trails a multiple of ATR below the highest high
of the lookback for long positions, and above the lowest low for short
positions. The name comes from the stop "hanging" from the extreme like a
chandelier. As the extreme ratchets in the trend direction, the exit trails;
when price crosses the exit level, the trend may be over.

## Recurrence

Let \(C_t, H_t, L_t\) be close, high, low; \(n\) be `window` (default \(22\));
and \(m\) be `multiplier` (default \(3.0\)).

\[
H^{\max}_t = \max_{0\le i < n} H_{t-i}, \qquad
L^{\min}_t = \min_{0\le i < n} L_{t-i}
\]

\[
A_t = \operatorname{ATR}_n(C_t, H_t, L_t)
\]

\[
\begin{aligned}
\operatorname{long\_exit}_t &= H^{\max}_t - m\, A_t \\
\operatorname{short\_exit}_t &= L^{\min}_t + m\, A_t
\end{aligned}
\]

Rolling high uses a max extreme; rolling low uses a min extreme. Nested ATR
inherits the outer `fillna` flag.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ChandelierExit` with `RollingExtreme` highs/lows and `ATR`. Result fields
are `long_exit` and `short_exit`. See also [`ATR`](atr.md).

## Reference

- [ChartSchool: Chandelier Exit](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chandelier-exit)
- [Investopedia: Chandelier Exit](https://www.investopedia.com/articles/trading/07/chandelier-exit.asp)
