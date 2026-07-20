# AccelerationBands

## Summary

`AccelerationBands` is RTTA's streaming Price Headley acceleration bands: upper
and lower envelopes formed by simple averages of range-scaled highs and lows,
with a middle SMA of close.

## Update API

```python
result = rtta.AccelerationBands(window=20, factor=4.0, fillna=True).update(close, high, low)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window`  | `20`    | SMA length for upper, lower, and middle |
| `factor`  | `4.0`   | Range-scale factor \(f\) |
| `fillna`  | `True`  | If `False`, NaN until `window` samples |

`update(...)` returns a result with fields `middle`, `upper`, `lower`.
`advance(close, high, low)` updates state; `last()` returns the cached result.

## Theory Of Operation

Each bar's high and low are expanded or contracted by a fraction of the bar
range relative to the bar's mid-range \(h+l\):

\[
s_t = f \frac{h_t - l_t}{h_t + l_t}.
\]

The upper source is \(h_t(1+s_t)\) and the lower source is \(l_t(1-s_t)\). Wide
range bars push the sources farther apart (more "acceleration"); quiet bars pull
them in. Independent SMAs of those sources form the plotted bands; the middle
band is a plain SMA of close for reference.

Breaks above the upper band or below the lower band are often read as
momentum extremes; tags of the middle can act as mean-reversion context.

## Recurrence

Let \(c_t,h_t,l_t\) be close, high, low; \(n\) the window; \(f\) the factor.

\[
s_t = f \cdot \frac{h_t - l_t}{h_t + l_t}
\]

(division is safe-divided in C++ when \(h_t+l_t=0\)).

\[
u^{src}_t = h_t(1 + s_t),\qquad
\ell^{src}_t = l_t(1 - s_t)
\]

\[
U_t = \operatorname{SMA}_n(u^{src}_t),\qquad
L_t = \operatorname{SMA}_n(\ell^{src}_t),\qquad
M_t = \operatorname{SMA}_n(c_t)
\]

Result: `middle` \(= M_t\), `upper` \(= U_t\), `lower` \(= L_t\).

When `fillna=False` and fewer than \(n\) samples have been seen, all three
fields are NaN. Upper/lower SMAs always use `fillna=True` internally so their
partial averages exist; the outer `fillna` gate controls the returned struct.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class AccelerationBands`.
- Result type: `AccelerationBandsResult` (`middle`, `upper`, `lower`).
- Three independent `SMA` instances: `upper_sma_`, `lower_sma_`, `middle_sma_`.
- Batch helper: `batch_acceleration_bands`.

## Reference

- [TradingView — Acceleration Bands](https://www.tradingview.com/support/solutions/43000589125-acceleration-bands/)
- [Price Headley acceleration bands overview](https://www.investopedia.com/terms/a/acceleration-bands.asp)
