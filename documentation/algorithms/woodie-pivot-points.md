# WoodiePivotPoints

## Summary

`WoodiePivotPoints` is RTTA's streaming Woodie pivot set. Levels for the current
bar are computed from the previous bar's high, low, and close with double weight
on close: \(PP = (H + L + 2C)/4\).

## Update API

```python
result = rtta.WoodiePivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

The first bar only seeds previous HLC. With `fillna=True`, all seven fields
return that bar's close; with `fillna=False`, they return `NaN`.

## Theory Of Operation

Woodie pivots place more weight on the previous close than classic floor pivots.
Support and resistance levels are then derived from that pivot and the previous
range, similar to classic R1/S1/R2/S2 formulas, with R3/S3 following the same
extension pattern as RTTA's classic `PivotPoints`.

## Recurrence

Let \(H_{t-1}, L_{t-1}, C_{t-1}\) be the previous bar's high, low, and close,
and let \(R = H_{t-1} - L_{t-1}\).

\[
PP_t = \frac{H_{t-1} + L_{t-1} + 2 C_{t-1}}{4}
\]

\[
\begin{aligned}
R1_t &= 2\, PP_t - L_{t-1} \\
S1_t &= 2\, PP_t - H_{t-1} \\
R2_t &= PP_t + R \\
S2_t &= PP_t - R \\
R3_t &= H_{t-1} + 2\,(PP_t - L_{t-1}) \\
S3_t &= L_{t-1} - 2\,(H_{t-1} - PP_t)
\end{aligned}
\]

After emitting levels, RTTA sets previous HLC to the current bar's \(H_t, L_t, C_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class WoodiePivotPoints`. The result type is `PivotPointsResult` with fields
`pp`, `r1`, `r2`, `r3`, `s1`, `s2`, `s3`.

## Reference

- [Investopedia: Woodie Pivot Points](https://www.investopedia.com/articles/technical/04/041404.asp)
- [ChartSchool: Pivot Points](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/pivot-points)
