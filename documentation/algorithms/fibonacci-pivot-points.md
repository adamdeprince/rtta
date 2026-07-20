# FibonacciPivotPoints

## Summary

`FibonacciPivotPoints` is RTTA's streaming Fibonacci pivot set. Levels for the
current bar come from the previous bar's high, low, and close: a classic mid
pivot plus support/resistance at \(0.382\), \(0.618\), and \(1.0\) of the
previous range.

## Update API

```python
result = rtta.FibonacciPivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

The first bar only seeds previous HLC. With `fillna=True`, all seven fields
return that bar's close; with `fillna=False`, they return `NaN`.

## Theory Of Operation

Fibonacci pivots combine the classic central pivot with Fibonacci retracement
ratios applied to the prior session's range. Traders treat R1–R3 / S1–S3 as
intraday support and resistance. RTTA computes levels from the prior completed
bar only, then advances previous HLC to the current bar — the same streaming
pattern as classic, Woodie, and Camarilla pivots.

## Recurrence

Let \(H_{t-1}, L_{t-1}, C_{t-1}\) be the previous bar's high, low, and close,
and let \(R = H_{t-1} - L_{t-1}\).

\[
PP_t = \frac{H_{t-1} + L_{t-1} + C_{t-1}}{3}
\]

\[
\begin{aligned}
R1_t &= PP_t + 0.382\, R \\
R2_t &= PP_t + 0.618\, R \\
R3_t &= PP_t + 1.000\, R \\
S1_t &= PP_t - 0.382\, R \\
S2_t &= PP_t - 0.618\, R \\
S3_t &= PP_t - 1.000\, R
\end{aligned}
\]

After emitting levels, RTTA sets previous HLC to the current bar's \(H_t, L_t, C_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class FibonacciPivotPoints`. The result type is `PivotPointsResult` with fields
`pp`, `r1`, `r2`, `r3`, `s1`, `s2`, `s3`.

## Reference

- [Investopedia: Fibonacci Pivot Points](https://www.investopedia.com/articles/technical/04/041404.asp)
- [ChartSchool: Pivot Points](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/pivot-points)
