# MACDFix

## Summary

`MACDFix` is RTTA's fixed-period MACD wrapper: EMA periods are locked at the
classic 12/26 fast/slow lengths while the signal period remains configurable.
It is a thin convenience layer over [`MACD`](macd.md).

## Update API

```python
result = rtta.MACDFix(signal=9, fillna=False).update(close)
```

| Parameter | Default  | Meaning |
|-----------|----------|---------|
| `signal`  | `9`      | Signal EMA length \(c\) |
| `fillna`  | `False`  | If `False`, NaN until warm-up |

`update(close)` returns `macd`, `signal`, `histogram` via the underlying
`MACD` result struct. `advance(close)` and `last()` delegate to the inner MACD.

## Theory Of Operation

Many charting APIs expose "MACD Fix" as MACD with non-adjustable 12/26 averages
(signal still free). RTTA implements that by constructing

```text
MACD(a=12, b=26, c=signal, fillna=fillna)
```

and forwarding every update. Numerically and semantically the indicator is
identical to calling `MACD(12, 26, signal, fillna)` directly.

## Recurrence

With fixed \(a=12\), \(b=26\), and configurable signal length \(c\):

\[
F_t = \operatorname{EMA}_{12}(x_t),\qquad
S_t = \operatorname{EMA}_{26}(x_t)
\]

\[
MACD_t = F_t - S_t
\]

\[
signal_t = \operatorname{EMA}_{c}(MACD_t)
\]

\[
hist_t = MACD_t - signal_t
\]

Warm-up when `fillna=False` matches `MACD`: NaN until \(\max(12,26)+c = 26+c\)
samples. Nested EMAs use `fillna=True` so state is always defined.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class MACDFix` as a member
  `MACD macd_(12, 26, signal, fillna)`.
- Batch helper: `batch_macd_fix`.
- See also [`MACD`](macd.md) and [`MACDExt`](macd-ext.md).

## Reference

- [StockCharts — MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
- [TA-Lib MACDFIX](https://ta-lib.org/functions/macdfix)
