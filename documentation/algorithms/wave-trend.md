# WaveTrend

## Summary

`WaveTrend` is RTTA's streaming LazyBear WaveTrend oscillator: a double-smoothed
channel index of HLC3 that produces `wt1` / `wt2` lines for overbought/oversold
and cross signals.

## Update API

```python
result = rtta.WaveTrend(
    channel_length=10, average_length=21, signal_length=4, fillna=True,
).update(high, low, close)
```

| Parameter         | Default | Meaning |
|-------------------|---------|---------|
| `channel_length`  | `10`    | EMA length for ESA and absolute deviation |
| `average_length`  | `21`    | EMA length for WT1 |
| `signal_length`   | `4`     | SMA length for WT2 signal |
| `fillna`          | `True`  | If `False`, NaN until combined warm-up |

`update(high, low, close)` returns `wt1`, `wt2`.
`advance(...)` updates state; `last()` returns the cached result.

## Theory Of Operation

WaveTrend is closely related to a commodity channel index on typical price, with
extra exponential smoothing:

1. Typical price \(ap = (h+l+c)/3\).
2. ESA: EMA of \(ap\).
3. Mean absolute deviation of \(ap\) from ESA, also EMA-smoothed.
4. Channel index \(ci = (ap - esa)/(0.015\cdot d)\).
5. WT1: EMA of \(ci\); WT2: SMA of WT1 (default 4).

Crosses of WT1 through WT2, and extreme levels (commonly near ±60 / ±53 in
LazyBear charts), are used for mean-reversion and momentum timing. The constant
`0.015` matches the LazyBear / TradingView reference scaling.

## Recurrence

\[
ap_t = \frac{h_t + l_t + c_t}{3}
\]

\[
esa_t = \operatorname{EMA}_{n_1}(ap_t)
\]

\[
d_t = \operatorname{EMA}_{n_1}\big(|ap_t - esa_t|\big)
\]

\[
ci_t = \frac{ap_t - esa_t}{0.015\, d_t}
\]

(safe division when \(d_t=0\)).

\[
wt1_t = \operatorname{EMA}_{n_2}(ci_t)
\]

\[
wt2_t = \operatorname{SMA}_{n_s}(wt1_t)
\]

Defaults: \(n_1=10\), \(n_2=21\), \(n_s=4\).

When `fillna=False`, require
\(n_1 + n_2 + n_s\) samples before non-NaN results (conservative warm-up matching
the C++ `warm_` sum).

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class WaveTrend`.
- Internals: `EMA esa_`, `EMA d_`, `EMA wt1_`, `SMA wt2_` (all with
  `fillna=True` on nested smoothers).
- Result type: `WaveTrendResult` (`wt1`, `wt2`).
- Batch helper: `batch_wave_trend`.

## Reference

- [TradingView — LazyBear WaveTrend Oscillator (WT)](https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/)
- [WaveTrend community documentation](https://www.tradingview.com/scripts/wavetrend/)
