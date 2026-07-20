# MACD

## Summary

`MACD` is RTTA's multi-output Moving Average Convergence/Divergence oscillator:
fast EMA minus slow EMA, a signal EMA of that difference, and a histogram of
MACD minus signal.

## Update API

```python
result = rtta.MACD(a=12, b=26, c=9, fillna=False).update(value)
```

| Parameter | Default  | Meaning |
|-----------|----------|---------|
| `a`       | `12`     | Fast EMA period |
| `b`       | `26`     | Slow EMA period |
| `c`       | `9`      | Signal EMA period |
| `fillna`  | `False`  | If `False`, NaN until warm-up |

`update(value)` returns a result with:

- `macd` — fast EMA − slow EMA
- `signal` — EMA of `macd`
- `histogram` — `macd` − `signal`

`advance(value)` updates state; `last()` returns the cached result.

Related APIs: [`MACDFix`](macd-fix.md) (fixed 12/26), [`MACDExt`](macd-ext.md)
(selectable SMA/EMA types).

## Theory Of Operation

MACD measures the gap between a short-horizon and long-horizon exponential
average of price. When the fast EMA is above the slow EMA, intermediate momentum
is positive; the opposite signals negative momentum. The signal line is a further
EMA that smooths MACD so that:

- **MACD / signal crosses** are common entry triggers.
- **Histogram** visualizes the gap and its expansion/contraction.
- **Zero-line crosses** of MACD mark shifts in the fast/slow EMA order.

All three nested EMAs run with internal `fillna=True` so coefficients and seeds
advance every bar; the outer `fillna` flag only gates whether incomplete
warm-up samples are returned as NaN.

## Recurrence

Let \(x_t\) be the input series and \(a,b,c\) the three periods.

\[
F_t = \operatorname{EMA}_a(x_t),\qquad
S_t = \operatorname{EMA}_b(x_t)
\]

\[
MACD_t = F_t - S_t
\]

\[
signal_t = \operatorname{EMA}_c(MACD_t)
\]

\[
hist_t = MACD_t - signal_t
\]

When `fillna=False`, all three fields are NaN while the sample counter is less
than \(\max(a,b)+c\); otherwise the current MACD triple is returned.

EMA seeding and \(\alpha=2/(n+1)\) follow RTTA's [`EMA`](ema.md).

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class MACD`.
- Members: `EMA a_`, `EMA b_`, `EMA c_`; result `MACDResult`.
- Default `fillna=False` (unlike many other RTTA indicators that default True).
- Batch helpers exist for MACD family series processing.

## Reference

- [StockCharts — MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
- [Investopedia — MACD](https://www.investopedia.com/terms/m/macd.asp)
