# PrettyGoodOscillator

## Summary

`PrettyGoodOscillator` is RTTA's streaming Mark Johnson Pretty Good Oscillator
(PGO): close minus its SMA, normalized by ATR over the same window.

## Update API

```python
value = rtta.PrettyGoodOscillator(window=14, fillna=True).update(close, high, low)
```

With `fillna=False`, output is `NaN` until `window` samples have been seen
(ATR's fillna also applies).

## Theory Of Operation

PGO asks how many ATRs the close sits away from its simple mean. Positive
readings mean the market is extended above the average; negative readings mean
it is extended below. Thresholds such as \(\pm 3\) are sometimes used for
exhaustion or mean-reversion setups. Because ATR is always non-negative, the
sign of PGO is determined only by \(close - SMA\).

## Recurrence

Let \(C_t, H_t, L_t\) be close, high, low and \(n\) be `window` (default \(14\)).

\[
S_t = \operatorname{SMA}_n(C_t), \qquad
A_t = \operatorname{ATR}_n(C_t, H_t, L_t)
\]

\[
PGO_t = \frac{C_t - S_t}{A_t}
\quad\text{(safe divide)}
\]

The nested SMA is constructed with `fillna=True` so a partial mean is available
during warmup; ATR uses the outer `fillna` flag. The outer warm count is \(n\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class PrettyGoodOscillator`. See also [`ATR`](atr.md).

## Reference

- [TradingView: Pretty Good Oscillator (PGO)](https://www.tradingview.com/script/rNYgL8uA-Pretty-Good-Oscillator-PGO/)
- [Investopedia: Average True Range (ATR)](https://www.investopedia.com/terms/a/atr.asp)
