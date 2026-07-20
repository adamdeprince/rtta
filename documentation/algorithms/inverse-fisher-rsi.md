# InverseFisherRSI

## Summary

`InverseFisherRSI` is RTTA's streaming Ehlers inverse Fisher transform applied
to RSI. RSI is mapped to a roughly symmetric domain, smoothed with a WMA, then
passed through the inverse Fisher map to produce a sharp oscillator typically
in \((-1, 1)\).

## Update API

```python
value = rtta.InverseFisherRSI(rsi_window=5, wma_window=9, fillna=True).update(close)
```

With `fillna=False`, output is `NaN` until `rsi_window + wma_window` samples
have been processed.

## Theory Of Operation

John Ehlers' inverse Fisher transform amplifies values near zero and compresses
extremes, turning a smooth oscillator into one with clearer turning points. The
common recipe for RSI is:

1. Compute RSI.
2. Scale RSI so mid-scale (50) maps near 0 and extremes map near \(\pm 5\).
3. Smooth with a short weighted moving average.
4. Apply \(\tanh\)-equivalent inverse Fisher on the smoothed value.

RTTA uses its streaming `RSI` and `WMA` primitives for steps 1–3.

## Recurrence

Let \(c_t\) be close, \(n_r\) be `rsi_window` (default \(5\)), and \(n_w\) be
`wma_window` (default \(9\)).

\[
RSI_t = \operatorname{RSI}_{n_r}(c_t)
\]

\[
x_t = 0.1\,(RSI_t - 50)
\]

\[
y_t = \operatorname{WMA}_{n_w}(x_t)
\]

\[
e_t = \exp(2 y_t), \qquad
IFR_t = \frac{e_t - 1}{e_t + 1}
\quad\text{(safe divide)}
\]

Note \(\frac{e^{2y}-1}{e^{2y}+1} = \tanh(y)\). Nested RSI/WMA are constructed
with `fillna=True`; the outer warm count is \(n_r + n_w\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class InverseFisherRSI`. See also [`RSI`](rsi.md).

## Reference

- [John F. Ehlers, "The Inverse Fisher Transform," *Stocks & Commodities*](https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf)
- [ChartSchool: Relative Strength Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi)
