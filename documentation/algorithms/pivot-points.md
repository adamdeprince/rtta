# PivotPoints

## Summary

`PivotPoints` is RTTA's streaming classic floor-trader pivot set. Levels for the
current bar are computed from the previous bar's high, low, and close; then the
current bar becomes the next previous bar.

## Update API

```python
result = rtta.PivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

The first bar only seeds previous HLC. With `fillna=True`, all seven fields
return that bar's close; with `fillna=False`, they return `NaN`.

## Theory Of Operation

Classic pivot points average the prior session's high, low, and close into a
central pivot \(PP\), then project resistance and support levels from that pivot
and the prior range. Floor traders historically used these as intraday reference
prices. RTTA's streaming form uses bar-to-bar previous HLC (suitable for any
regular bar stream), not a separate session calendar.

Related variants: [`WoodiePivotPoints`](woodie-pivot-points.md),
[`CamarillaPivotPoints`](camarilla-pivot-points.md),
[`FibonacciPivotPoints`](fibonacci-pivot-points.md).

## Recurrence

Let \(H_{t-1}, L_{t-1}, C_{t-1}\) be the previous bar's high, low, and close.

\[
PP_t = \frac{H_{t-1} + L_{t-1} + C_{t-1}}{3}
\]

\[
\begin{aligned}
R1_t &= 2\, PP_t - L_{t-1} \\
S1_t &= 2\, PP_t - H_{t-1} \\
R2_t &= PP_t + (H_{t-1} - L_{t-1}) \\
S2_t &= PP_t - (H_{t-1} - L_{t-1}) \\
R3_t &= H_{t-1} + 2\,(PP_t - L_{t-1}) \\
S3_t &= L_{t-1} - 2\,(H_{t-1} - PP_t)
\end{aligned}
\]

After emitting levels, RTTA sets previous HLC to the current bar's \(H_t, L_t, C_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class PivotPoints`. The result type is `PivotPointsResult` with fields
`pp`, `r1`, `r2`, `r3`, `s1`, `s2`, `s3`.

## Reference

- [ChartSchool: Pivot Points](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/pivot-points)
- [Investopedia: Pivot Point](https://www.investopedia.com/terms/p/pivotpoint.asp)
