# GuppyMultipleMovingAverage

## Summary

`GuppyMultipleMovingAverage` is RTTA's compact Guppy MMA: six short EMAs and six
long EMAs are updated each bar, then summarized as short-group average,
long-group average, and their spread.

## Update API

```python
result = rtta.GuppyMultipleMovingAverage(fillna=True).update(price)
# result.short_average, result.long_average, result.spread
```

Periods are fixed: short \(\{3,5,8,10,12,15\}\), long
\(\{30,35,40,45,50,60\}\). With `fillna=False`, outputs are `NaN` until 60
samples have been seen.

## Theory Of Operation

Daryl Guppy's Multiple Moving Average frames market participation as two
populations: short-term traders and longer-term investors. Averaging each
population's EMAs yields two summary curves; their separation (`spread`) is a
simple trend-strength / agreement measure. For the full twelve-line ribbon, use
[`GuppyMMARibbon`](guppy-mma-ribbon.md).

## Recurrence

Let \(x_t\) be price,

\[
\mathcal{S} = \{3,5,8,10,12,15\}, \qquad
\mathcal{L} = \{30,35,40,45,50,60\}.
\]

\[
\begin{aligned}
S^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{S}} \operatorname{EMA}_p(x_t) \\
L^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{L}} \operatorname{EMA}_p(x_t) \\
\operatorname{spread}_t &= S^{\text{avg}}_t - L^{\text{avg}}_t
\end{aligned}
\]

Each \(\operatorname{EMA}_p\) uses \(\alpha = 2/(p+1)\) as in RTTA's `class EMA`.
Nested EMAs always fill during warmup; the outer `fillna` only blanks the first
59 samples when false.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class GuppyMultipleMovingAverage`. Result fields are `short_average`,
`long_average`, and `spread`.

## Reference

- [Investopedia: Guppy Multiple Moving Average (GMMA)](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp)
- [ChartSchool: Guppy Multiple Moving Average](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/guppy-multiple-moving-average)
