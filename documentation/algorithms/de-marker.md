# DeMarker

## Summary

`DeMarker` is RTTA's streaming DeMarker oscillator. It compares rolling upward
high extensions to downward low extensions and maps them into a \(0\)–\(1\)
ratio.

## Update API

```python
value = rtta.DeMarker(window=14, fillna=True).update(high, low)
```

The first bar contributes zero DeMax/DeMin (no previous high/low). With
`fillna=False`, output is `NaN` until the window buffer is full.

## Theory Of Operation

Tom DeMark's DeMarker measures demand pressure via positive high-to-high changes
and supply pressure via negative low-to-low changes. Averaging each over a window
and taking the ratio yields an oscillator bounded in \([0, 1]\). Readings near
\(1\) indicate persistent upward high extensions; near \(0\) indicates
persistent downward low extensions. Extreme zones (classically near \(0.7\) /
\(0.3\)) are used as overbought/oversold references.

## Recurrence

Let \(H_t, L_t\) be high and low and \(n\) be `window` (default \(14\)).

\[
DeMax_t = \max(H_t - H_{t-1}, 0), \qquad
DeMin_t = \max(L_{t-1} - L_t, 0)
\]

(with \(DeMax_0 = DeMin_0 = 0\)). Maintain rolling sums over the last
\(\min(t+1, n)\) samples:

\[
M_t = \sum_{i \in W_t} DeMax_i, \qquad
N_t = \sum_{i \in W_t} DeMin_i
\]

\[
DeM_t = \frac{M_t}{M_t + N_t}
\quad\text{(safe divide)}
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class DeMarker`
using two rolling sum buffers (`demax_`, `demin_`).

## Reference

- [Investopedia: DeMarker Indicator](https://www.investopedia.com/terms/d/demarkerindicator.asp)
- [TradingPedia: DeMarker](https://www.tradingpedia.com/forex-trading-indicators/demarker-indicator/)
