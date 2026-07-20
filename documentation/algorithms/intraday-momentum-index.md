# IntradayMomentumIndex

## Summary

`IntradayMomentumIndex` is RTTA's streaming Intraday Momentum Index (IMI): an
RSI-style oscillator of open-to-close gains versus losses within each bar,
aggregated over a rolling window.

## Update API

```python
result = rtta.IntradayMomentumIndex(window=14, fillna=True).update(open, close)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window`  | `14`    | Rolling sum window for gains and losses |
| `fillna`  | `True`  | If `False`, NaN until the window is full |

`update(open, close)` returns a scalar IMI value in \([0, 100]\) when both
rolling sums are defined.

## Theory Of Operation

Classic RSI uses close-to-close changes. IMI instead classifies each bar by its
**intrabar** direction:

- If \(close > open\), the full open-to-close rise is a gain.
- If \(close < open\), the full open-to-close fall is a loss.
- Unchanged open/close contributes zero gain and zero loss.

Rolling sums of gains and losses over \(n\) bars form an RSI-like ratio. High
IMI means most recent bars closed above their opens (intraday buying pressure);
low IMI means the opposite. It is often used on intraday bars or as a day-session
sentiment measure on daily OHLC.

Unlike Wilder RSI, this implementation uses **simple rolling sums** (not Wilder
smoothing) of the gain/loss series.

## Recurrence

Let \(o_t,c_t\) be open and close, \(n\) the window.

\[
G_t =
\begin{cases}
c_t - o_t & c_t > o_t \\
0 & \text{otherwise}
\end{cases}
,\qquad
L_t =
\begin{cases}
o_t - c_t & c_t < o_t \\
0 & \text{otherwise}
\end{cases}
\]

\[
IMI_t = 100 \cdot \frac{\sum_{i=0}^{n-1} G_{t-i}}{\sum_{i=0}^{n-1} G_{t-i} + \sum_{i=0}^{n-1} L_{t-i}}
\]

When `fillna=True`, partial windows use the samples accumulated so far. When
`fillna=False`, return NaN until the gain buffer is full. Safe division yields a
defined value when both sums are zero.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class IntradayMomentumIndex`.
- Two `RollingBuffer`s with running `gain_sum_` / `loss_sum_` via
  `rolling_sum_push`.
- Output is a scalar `double`, not a result struct.

## Reference

- [Investopedia — Intraday Momentum Index (IMI)](https://www.investopedia.com/terms/i/intraday-momentum-index-imi.asp)
- [IMI overview](https://www.tradingview.com/support/solutions/43000589111-intraday-momentum-index/)
