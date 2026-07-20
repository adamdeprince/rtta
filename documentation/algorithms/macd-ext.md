# MACDExt

## Summary

`MACDExt` is RTTA's extended MACD: the same macd / signal / histogram structure
as classic MACD, but with independently selectable SMA or EMA smoothers for the
fast, slow, and signal stages (TA-Lib `MACDEXT` style).

## Update API

```python
result = rtta.MACDExt(
    fast=12, slow=26, signal=9,
    fast_ma_type=1, slow_ma_type=1, signal_ma_type=1,
    fillna=True,
).update(value)
```

| Parameter         | Default | Meaning |
|-------------------|---------|---------|
| `fast`            | `12`    | Fast MA length |
| `slow`            | `26`    | Slow MA length |
| `signal`          | `9`     | Signal MA length |
| `fast_ma_type`    | `1`     | `0` = SMA, `1` = EMA |
| `slow_ma_type`    | `1`     | `0` = SMA, `1` = EMA |
| `signal_ma_type`  | `1`     | `0` = SMA, `1` = EMA |
| `fillna`          | `True`  | If `False`, NaN until warm-up |

`update(value)` returns `macd`, `signal`, `histogram` (same fields as
[`MACD`](macd.md)). `advance(value)` updates state; `last()` returns the cache.

## Theory Of Operation

Classic MACD uses EMAs throughout. Some platforms and TA-Lib allow mixing SMA
and EMA so users can, for example, use SMA signal lines or all-SMA MACD for
research. `MACDExt` implements that flexibility with a small `SelectableMA`
helper (`0→SMA`, anything else→EMA).

Interpretation is unchanged: MACD is the fast−slow spread, signal is a MA of
that spread, histogram is MACD−signal. Zero-line and signal crosses retain the
usual momentum meaning.

## Recurrence

Let \(x_t\) be the input. Let \(\operatorname{MA}^{(type)}_n\) be SMA if
`type=0` and EMA if `type=1`.

\[
F_t = \operatorname{MA}^{(f)}_{n_f}(x_t),\qquad
S_t = \operatorname{MA}^{(s)}_{n_s}(x_t)
\]

\[
MACD_t = F_t - S_t
\]

\[
signal_t = \operatorname{MA}^{(g)}_{n_g}(MACD_t)
\]

\[
hist_t = MACD_t - signal_t
\]

Nested MAs always run with `fillna=True` so intermediate values exist. When
outer `fillna=False`, all three outputs are NaN until
\(\max(n_f,n_s)+n_g\) samples have been processed.

With all `ma_type=1` and default lengths, results match EMA-based MACD aside
from warm-up / `fillna` conventions.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class MACDExt` and
  `class SelectableMA`.
- Result type: shared `MACDResult` (`macd`, `signal`, `histogram`).
- Batch helper: `batch_macd_ext`.
- Only SMA and EMA are selectable in this build (not DEMA/TEMA/etc. from full
  TA-Lib MA type enums).

## Reference

- [TA-Lib MACDEXT](https://ta-lib.org/functions/macdext)
- [StockCharts — MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
