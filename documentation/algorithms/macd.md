# MACD

## Summary

`MACD` is RTTA's scalar Moving Average Convergence/Divergence pipeline. It
computes a fast EMA, a slow EMA, subtracts them, and returns an EMA-smoothed
signal value.

## Update API

```python
value = rtta.MACD(a=12, b=26, c=9, fillna=False).update(value)
```

`a` is the fast EMA window, `b` is the slow EMA window, and `c` is the signal EMA
window.

## Theory Of Operation

MACD measures the distance between a fast and slow trend estimate. RTTA's
scalar `MACD` class returns the signal-smoothed line:

- fast EMA of the input,
- slow EMA of the input,
- raw difference between the two,
- signal EMA of that raw difference.

For a multi-field percentage oscillator with signal and histogram fields, see
`PercentagePrice`.

## Recurrence

Let \(x_t\) be the input, and define EMA recurrences with smoothing constants
\(\alpha_a=2/(1+a)\), \(\alpha_b=2/(1+b)\), and \(\alpha_c=2/(1+c)\).

\[
F_t = \alpha_a x_t + (1-\alpha_a)F_{t-1}
\]

\[
S_t = \alpha_b x_t + (1-\alpha_b)S_{t-1}
\]

\[
D_t = F_t - S_t
\]

\[
M_t = \alpha_c D_t + (1-\alpha_c)M_{t-1}
\]

`update(...)` returns \(M_t\). With `fillna=False`, returned values are `NaN`
until the internal counter reaches \(\max(a,b)+c\), while all EMA state still
advances.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MACD`.

## Reference

- [ChartSchool: MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
