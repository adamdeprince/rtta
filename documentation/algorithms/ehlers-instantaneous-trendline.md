# EhlersInstantaneousTrendline

## Summary

`EhlersInstantaneousTrendline` is RTTA's streaming Ehlers instantaneous
trendline: a two-pole recursive smoother with a four-sample input average and
a two-bar extrapolated **trigger** line for timing.

## Update API

```python
result = rtta.EhlersInstantaneousTrendline(period=20, fillna=True).update(price)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `period`  | `20`    | Critical period \(P\) (minimum 2) |
| `fillna`  | `True`  | If `False`, NaN until `period` samples |

`update(...)` returns:

- `trendline` — instantaneous trendline
- `trigger` — \(2\cdot\text{trendline}_t - \text{trendline}_{t-2}\)

`advance(price)` updates state; `last()` returns the cached result.

## Theory Of Operation

Ehlers' instantaneous trendline is a low-lag smooth of price intended to track
the local mean of the cycle component without waiting for a long moving-average
window. Coefficients match the Super Smoother pole placement
(\(\sqrt{2}\,\pi / P\)), but the feed-forward gain is distributed as
\(c_1 = (1-c_2-c_3)/4\) over a three-term average of price
\((x_t + 2x_{t-1} + x_{t-2})\), which is the discrete form used in Ehlers'
iTrend presentations.

The **trigger** is a two-bar linear extrapolation of the trendline
(\(2 y_t - y_{t-2}\)). Crosses of trendline and trigger mark short-term turns
of the smooth, analogous to a signal line without an extra EMA.

This indicator is distinct from TA-Lib / RTTA
[`HilbertTrendline`](hilbert-trendline.md), which adapts the averaging length
from a Hilbert dominant-cycle estimate rather than a fixed period.

## Recurrence

Let \(P = \max(\texttt{period}, 2)\). Precompute:

\[
\theta = \frac{\sqrt{2}\,\pi}{P},\qquad
a_1 = e^{-\theta},\qquad
b_1 = 2 a_1 \cos(\theta)
\]

\[
c_2 = b_1,\qquad
c_3 = -a_1^{2},\qquad
c_1 = \frac{1 - c_2 - c_3}{4}
\]

For the first two samples:

\[
I_t = x_t
\]

Thereafter:

\[
I_t = c_1\,(x_t + 2 x_{t-1} + x_{t-2}) + c_2\, I_{t-1} + c_3\, I_{t-2}
\]

Trigger (using trend state **before** shifting; after update, \(t2\) is the
prior prior trend):

\[
Trig_t = 2 I_t - I_{t-2}
\]

In code: `trigger = 2.0 * trend - t2_` where `t2_` is still the old
two-bar-ago trendline at computation time, then state shifts
`t2_ ← t1_ ← trend`.

Result: `trendline` \(= I_t\), `trigger` \(= Trig_t\).

When `fillna=False` and fewer than \(P\) samples have been processed, both
fields are NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class EhlersInstantaneousTrendline`.
- Result type: `EhlersInstantaneousTrendlineResult` (`trendline`, `trigger`).
- Price lags `p1_`,`p2_`; trend lags `t1_`,`t2_`.
- Batch helper: `batch_ehlers_itrend`.

## Reference

- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
- [Instantaneous Trendline / Super Smoother material](https://www.mesasoftware.com/papers/ZeroLag.pdf)
