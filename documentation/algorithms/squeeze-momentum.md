# SqueezeMomentum

## Summary

`SqueezeMomentum` is RTTA's streaming TTM-style squeeze indicator: a binary flag
for Bollinger bands inside Keltner channels (volatility compression) plus a
linear-regression momentum of price relative to a hybrid mid-line.

## Update API

```python
result = rtta.SqueezeMomentum(window=20, bb_mult=2.0, kc_mult=1.5, fillna=True).update(close, high, low)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window`  | `20`    | Lookback for BB, ATR/KC, Donchian, and linreg |
| `bb_mult` | `2.0`   | Bollinger standard-deviation multiplier |
| `kc_mult` | `1.5`   | Keltner ATR multiplier |
| `fillna`  | `True`  | If `False`, NaN until `window` samples |

`update(...)` returns:

- `on` — `1.0` if squeeze is on (BB inside KC), else `0.0`
- `momentum` — linear-regression value of the de-meaned series

`advance(...)` updates state; `last()` returns the cached result.

## Theory Of Operation

**Squeeze on** means Bollinger bands (mean ± \(k\cdot\sigma\)) lie entirely
inside Keltner channels (mean ± \(m\cdot ATR\)). That is a classic TTM Squeeze
compression regime: realized volatility (BB) is low relative to average true
range width (KC). When the squeeze "fires" (`on` flips from 1 to 0), energy is
often released as a directional move.

**Momentum** measures where close sits versus a basis that averages the Donchian
midpoint and the SMA of close, then fits a rolling linear regression (as in
LazyBear / TTM-style histograms). Positive momentum favors upside release;
negative favors downside.

## Recurrence

Let \(n\) be the window, \(k=\texttt{bb\_mult}\), \(m=\texttt{kc\_mult}\).

Rolling mean and population stdev of close:

\[
\mu_t = \operatorname{mean}_n(c),\qquad
\sigma_t = \operatorname{stddev}_n(c)
\]

\[
BB^{hi}_t = \mu_t + k\sigma_t,\qquad
BB^{lo}_t = \mu_t - k\sigma_t
\]

With ATR of length \(n\):

\[
KC^{hi}_t = \mu_t + m\cdot ATR_t,\qquad
KC^{lo}_t = \mu_t - m\cdot ATR_t
\]

Squeeze flag:

\[
on_t =
\begin{cases}
1 & BB^{lo}_t > KC^{lo}_t \;\land\; BB^{hi}_t < KC^{hi}_t \\
0 & \text{otherwise}
\end{cases}
\]

Donchian mid and hybrid basis:

\[
HH_t = \max_{0\le i<n} h_{t-i},\qquad
LL_t = \min_{0\le i<n} l_{t-i}
\]

\[
basis_t = \frac{1}{2}\left(\frac{HH_t+LL_t}{2} + \mu_t\right)
\]

\[
\delta_t = c_t - basis_t
\]

\[
momentum_t = \operatorname{LinReg}_n(\delta)_t
\]

(value of the regression fit at the current bar, via `LinearRegressionCore`).

When `fillna=False` and fewer than \(n\) samples have been seen, both fields are
NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class SqueezeMomentum`.
- Uses rolling sum/sum2 for mean/variance, `RollingExtreme` for HH/LL,
  `ATR`, and `LinearRegressionCore linreg_`.
- Result type: `SqueezeMomentumResult` (`on`, `momentum`).
- Batch helper: `batch_squeeze_momentum`.
- Note: Keltner mid uses the SMA of close (`mean`), not a separate EMA of
  typical price—matching this C++ implementation.

## Reference

- [StockCharts — TTM Squeeze](https://school.stockcharts.com/doku.php?id=technical_indicators:ttm_squeeze)
- [LazyBear Squeeze Momentum (TradingView community)](https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/)
