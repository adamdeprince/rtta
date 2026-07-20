# TrendIntensityIndex

## Summary

`TrendIntensityIndex` is RTTA's streaming Trend Intensity Index (TII): the
percentage of absolute close-to-SMA deviations that are positive over a rolling
window, scaled to \(0\)–\(100\).

## Update API

```python
value = rtta.TrendIntensityIndex(window=30, fillna=True).update(close)
```

With `fillna=False`, output is `NaN` until `window` samples have been seen.

## Theory Of Operation

TII measures how consistently price sits above its moving average. For each bar
the signed residual \(close - SMA\) is split into a positive part and its
absolute value; rolling sums of those quantities form a ratio. A reading near
100 means price has been almost exclusively above the SMA (strong uptrend
intensity); near 0 means almost exclusively below; near 50 means a balanced
mix.

## Recurrence

Let \(c_t\) be close and \(n\) be `window` (default \(30\)).

\[
S_t = \operatorname{SMA}_n(c_t), \qquad
d_t = c_t - S_t
\]

\[
p_t = \max(d_t, 0), \qquad
a_t = |d_t|
\]

Maintain rolling sums over the last \(\min(t, n)\) samples:

\[
P_t = \sum_{i \in W_t} p_i, \qquad
A_t = \sum_{i \in W_t} a_i
\]

\[
TII_t = 100 \cdot \frac{P_t}{A_t}
\quad\text{(safe divide)}
\]

The nested SMA is constructed with `fillna=True`. Outer warm count is \(n\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class TrendIntensityIndex` with members `sma_`, `pos_`, and `abs_`.

## Reference

- [TradingView: Trend Intensity Index](https://www.tradingview.com/script/uCvHH824-Trend-Intensity-Index/)
- [Investopedia: Trend](https://www.investopedia.com/terms/t/trend.asp)
