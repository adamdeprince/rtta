# RandomWalkIndex

## Summary

`RandomWalkIndex` is RTTA's streaming Random Walk Index (RWI). It compares the
current high/low extremes against the window's opposite extreme, scaled by ATR
times \(\sqrt{n}\), yielding separate high-side and low-side readings.

## Update API

```python
result = rtta.RandomWalkIndex(window=14, fillna=True).update(close, high, low)
# result.high, result.low
```

With `fillna=False`, both fields are `NaN` until `window` samples have been
seen.

## Theory Of Operation

The Random Walk Index asks whether price has moved farther than a random walk
would typically travel over \(n\) bars. Under a simple diffusion model, expected
distance grows like \(\sigma\sqrt{n}\); ATR stands in for \(\sigma\). When
RWI High is large, the market has advanced more than random-walk noise from the
window low; when RWI Low is large, it has declined more than noise from the
window high. Values above roughly 1 are often interpreted as non-random trend
evidence on that side.

## Recurrence

Let \(C_t, H_t, L_t\) be close, high, low and \(n\) be `window` (default \(14\)).

Maintain a rolling maximum of highs and rolling minimum of lows over \(n\) bars,
and Wilder ATR over the same length:

\[
H^{\max}_t = \max_{0\le i < n} H_{t-i}, \qquad
L^{\min}_t = \min_{0\le i < n} L_{t-i}
\]

\[
A_t = \operatorname{ATR}_n(C_t, H_t, L_t)
\]

\[
\operatorname{scale}_t = A_t \sqrt{n}
\]

\[
\begin{aligned}
\operatorname{RWI\text{-}High}_t &= \frac{H_t - L^{\min}_t}{\operatorname{scale}_t} \\
\operatorname{RWI\text{-}Low}_t &= \frac{H^{\max}_t - L_t}{\operatorname{scale}_t}
\end{aligned}
\]

(safe divide). Result field `high` is RWI-High; field `low` is RWI-Low. Nested
ATR uses `fillna=True`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class RandomWalkIndex` using `RollingExtreme` highs/lows and `ATR`.

## Reference

- [Investopedia: Random Walk Index](https://www.investopedia.com/terms/r/random-walk-index.asp)
- [TradingPedia: Random Walk Index](https://www.tradingpedia.com/forex-trading-indicators/random-walk-index/)
