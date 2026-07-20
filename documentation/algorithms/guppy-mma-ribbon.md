# GuppyMMARibbon

## Summary

`GuppyMMARibbon` is RTTA's full Daryl Guppy Multiple Moving Average ribbon. It
tracks twelve EMAs (six short-term trader lengths and six long-term investor
lengths), their group averages, and the short-minus-long spread.

## Update API

```python
result = rtta.GuppyMMARibbon(fillna=True).update(price)
# short: result.s3, s5, s8, s10, s12, s15
# long:  result.l30, l35, l40, l45, l50, l60
# aggs:  result.short_average, result.long_average, result.spread
```

Periods are fixed at the classic Guppy lengths. With `fillna=False`, all fields
are `NaN` until 60 samples (longest EMA period) have been seen.

## Theory Of Operation

Guppy's MMAs separate "traders" (fast EMAs) from "investors" (slow EMAs). When
the short group is tightly bunched and pulls away from a similarly coherent long
group, trend conviction is high. Compression or interleaving of the groups
suggests indecision. The ribbon form exposes every EMA for charting, not only
the two averages.

RTTA also provides a compact sibling [`GuppyMultipleMovingAverage`](guppy-multiple-moving-average.md)
that returns only the averages and spread.

## Recurrence

Let \(x_t\) be price. Define fixed period sets

\[
\mathcal{S} = \{3,5,8,10,12,15\}, \qquad
\mathcal{L} = \{30,35,40,45,50,60\}.
\]

\[
E^{(p)}_t = \operatorname{EMA}_p(x_t)
\quad\text{for each } p \in \mathcal{S} \cup \mathcal{L}
\]

(RTTA's EMA uses multiplier \(\alpha = 2/(p+1)\).)

\[
\begin{aligned}
S^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{S}} E^{(p)}_t \\
L^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{L}} E^{(p)}_t \\
\operatorname{spread}_t &= S^{\text{avg}}_t - L^{\text{avg}}_t
\end{aligned}
\]

Named outputs map directly: `s3` \(= E^{(3)}_t\), …, `s15` \(= E^{(15)}_t\),
`l30` \(= E^{(30)}_t\), …, `l60` \(= E^{(60)}_t\), plus the three aggregates
above. Nested EMAs are constructed with `fillna=True` so partial values exist
during warmup; the outer `fillna` only gates emission of `NaN` for the first 59
bars when false.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class GuppyMMARibbon`. Result fields are `s3`–`s15`, `l30`–`l60`,
`short_average`, `long_average`, and `spread`.

## Reference

- [Investopedia: Guppy Multiple Moving Average (GMMA)](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp)
- [ChartSchool: Guppy Multiple Moving Average](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/guppy-multiple-moving-average)
