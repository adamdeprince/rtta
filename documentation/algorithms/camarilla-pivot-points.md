# CamarillaPivotPoints

## Summary

`CamarillaPivotPoints` is RTTA's streaming Camarilla pivot set. Levels for the
current bar are computed from the previous bar's high, low, and close using
Camarilla range multipliers, then the current bar becomes the next previous bar.

## Update API

```python
result = rtta.CamarillaPivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

The first bar only seeds previous HLC. With `fillna=True`, all seven fields
return that bar's close; with `fillna=False`, they return `NaN`.

## Theory Of Operation

Nick Scott's Camarilla equation places support/resistance tightly around the
previous close using fixed fractions of the previous range. The classic levels
use multipliers \(1.1/12\), \(1.1/6\), and \(1.1/4\) for R1/S1 through R3/S3.
RTTA also reports a classic mid pivot \(PP = (H+L+C)/3\) for convenience in the
same result struct used by floor/Woodie/Fib pivots.

Because levels depend only on the prior completed bar, the streaming form stores
previous HLC and recomputes once per bar before overwriting that previous state.

## Recurrence

Let \(H_{t-1}, L_{t-1}, C_{t-1}\) be the previous bar's high, low, and close,
and let \(R = H_{t-1} - L_{t-1}\).

\[
PP_t = \frac{H_{t-1} + L_{t-1} + C_{t-1}}{3}
\]

\[
\begin{aligned}
R1_t &= C_{t-1} + R \cdot \tfrac{1.1}{12} \\
R2_t &= C_{t-1} + R \cdot \tfrac{1.1}{6} \\
R3_t &= C_{t-1} + R \cdot \tfrac{1.1}{4} \\
S1_t &= C_{t-1} - R \cdot \tfrac{1.1}{12} \\
S2_t &= C_{t-1} - R \cdot \tfrac{1.1}{6} \\
S3_t &= C_{t-1} - R \cdot \tfrac{1.1}{4}
\end{aligned}
\]

After emitting levels, RTTA sets previous HLC to the current bar's \(H_t, L_t, C_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class CamarillaPivotPoints`. The result type is `PivotPointsResult` with fields
`pp`, `r1`, `r2`, `r3`, `s1`, `s2`, `s3`.

## Reference

- [Investopedia: Camarilla Pivot Points](https://www.investopedia.com/terms/c/camarilla-pivot-point.asp)
- [TradingPedia: Camarilla Pivot Points](https://www.tradingpedia.com/forex-trading-strategies/camarilla-pivot-points/)
