# HilbertTrendline

## Summary

`HilbertTrendline` is RTTA's streaming period-adaptive instantaneous trendline
(TA-Lib `HT_TRENDLINE`). It averages raw price over the current dominant-cycle
length, then applies a 4-bar weighted moving average of that average.

## Update API

```python
result = rtta.HilbertTrendline(fillna=True).update(value)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fillna`  | `True`  | If `False`, NaN until lookback of 63 samples is exceeded |

`update(value)` returns the current adaptive trendline as a scalar.

## Theory Of Operation

Where a fixed-period SMA always lags the same amount, the Hilbert trendline
shortens its averaging window when the dominant cycle is short and lengthens it
when the cycle is long. TA-Lib averages **raw** prices over
\(N=\operatorname{clip}(\lfloor\overline{P}+0.5\rfloor,1,50)\) bars (not the
WMA smooth), then smooths that average with the same 4-tap weights used on the
input price:

\[
(4 A_t + 3 A_{t-1} + 2 A_{t-2} + A_{t-3})/10.
\]

This is the adaptive counterpart to
[`EhlersInstantaneousTrendline`](ehlers-instantaneous-trendline.md), which uses
a fixed critical period and a two-pole recursion instead of a Hilbert period.

## Recurrence

Let \(\overline{P}_t\) be the smoothed dominant period and

\[
N = \operatorname{clip}\!\big(\lfloor \overline{P}_t + 0.5 \rfloor,\; 1,\; 50\big).
\]

Average of the last \(N\) raw prices (from a retained price history of up to 64
samples):

\[
A_t = \frac{1}{N}\sum_{j=0}^{N-1} x_{t-j}
\]

(missing history treated as zero contribution in early bars when fewer than \(N\)
prices exist).

Four-bar WMA of the averages:

\[
TL_t = \frac{4 A_t + 3 A_{t-1} + 2 A_{t-2} + A_{t-3}}{10}
\]

Returned value: \(TL_t\).

With `fillna=False`, output is NaN until more than 63 updates have completed.

## Implementation Notes

- Thin wrapper around `HilbertCycleEngine::trendline()` (`class HilbertTrendline`).
- Engine keeps `prices_` (max 64), and `i_trend1_`,`i_trend2_`,`i_trend3_` as
  lags of \(A_t\).
- Lookback: `lookback_phase_ = 63`.

## Reference

- [TA-Lib HT_TRENDLINE](https://ta-lib.org/functions/ht_trendline)
- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
