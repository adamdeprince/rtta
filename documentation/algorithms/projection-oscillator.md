# ProjectionOscillator

## Summary

`ProjectionOscillator` is RTTA's streaming projection oscillator: a
stochastic-style placement of close between linear-regression projection bands
of high and low, with a short SMA signal line.

## Update API

```python
result = rtta.ProjectionOscillator(
    window=14, signal_window=3, fillna=True
).update(high, low, close)
# result.value, result.signal, result.upper, result.lower
```

The `update(...)` call consumes `high`, `low`, and `close`. `advance(...)`
updates state without returning a Python object. Multi-output `batch(...)`
returns arrays for `value`, `signal`, `upper`, and `lower`.

## Theory Of Operation

Projection bands fit a rolling least-squares line to highs and another to lows.
The fitted values at the latest bar form a time-varying channel. Mapping close
into that channel as a percent (as a stochastic does with high/low extremes)
produces an oscillator that respects the local regression slope rather than raw
window extrema. The signal is a short SMA of the oscillator.

## Recurrence

Let \(h_t\), \(\ell_t\), \(c_t\) be high, low, and close, and let \(n\) be
`window`. Fit rolling linear regressions on the high and low series and take the
fitted values at the current end of the window (RTTA `LinearRegressionCore`
field `value`):

\[
U_t = \operatorname{LinReg}_n(h)_t, \qquad
L_t = \operatorname{LinReg}_n(\ell)_t
\]

\[
\operatorname{PO}_t = 100\cdot\frac{c_t - L_t}{U_t - L_t}
\]

\[
\operatorname{signal}_t = \operatorname{SMA}_{n_s}(\operatorname{PO}_t)
\]

where \(n_s\) is `signal_window`. Division by a zero band width is safe-divided
to \(0\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ProjectionOscillator` with two `LinearRegressionCore` members and one
`SMA` for the signal.

## Reference

- [Investopedia: Projection Oscillator](https://www.investopedia.com/terms/p/projectionoscillator.asp)
- [ChartSchool: Linear Regression](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
