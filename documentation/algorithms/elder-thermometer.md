# ElderThermometer

## Summary

`ElderThermometer` is RTTA's streaming Elder-style bar-range thermometer. Each
bar reports its high-low range, the ratio of that range to the previous bar's
range, and a binary "hot" flag when the current range exceeds the previous one.

## Update API

```python
result = rtta.ElderThermometer(fillna=True).update(high, low)
# result.ratio, result.hot, result.range
```

If `high < low`, the implementation swaps them so range is always non-negative.
The first bar seeds previous range; with `fillna=True` it returns
`ratio=1.0`, `hot=0.0`, and the current range; with `fillna=False` all fields
are `NaN` on the first bar.

## Theory Of Operation

Alexander Elder uses a market "thermometer" concept to compare today's activity
range with yesterday's. Expanding ranges ("hot" bars) often accompany trending
or news-driven sessions; contracting ranges suggest quieter conditions. RTTA
exposes three related outputs so callers can threshold ratio, use the binary
hot flag, or chart raw range.

## Recurrence

Let \(H_t, L_t\) be high and low (after optional swap so \(H_t \ge L_t\)).

\[
\rho_t = H_t - L_t
\]

On the first bar, store \(\rho_t\) as previous range and emit the fillna
placeholder. For \(t \ge 1\):

\[
\operatorname{ratio}_t = \frac{\rho_t}{\rho_{t-1}}
\quad\text{(safe divide)}
\]

\[
\operatorname{hot}_t =
\begin{cases}
1, & \rho_t > \rho_{t-1} \\
0, & \text{otherwise}
\end{cases}
\]

\[
\operatorname{range}_t = \rho_t
\]

Then set \(\rho_{t}\) as the new previous range.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ElderThermometer`. Result fields are `ratio`, `hot`, and `range`.

## Reference

- [Investopedia: Alexander Elder](https://www.investopedia.com/terms/a/alexander-elder.asp)
- [TradingView: Elder Thermometer concepts](https://www.tradingview.com/scripts/elder/)
