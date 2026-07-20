# AndrewsPitchfork

## Summary

`AndrewsPitchfork` is RTTA's streaming Andrews Pitchfork built from percent
ZigZag-style pivots. Once three alternating pivots are confirmed, each bar
emits the median line and parallel upper/lower tines evaluated at the current
index, plus pivot and direction flags.

## Update API

```python
result = rtta.AndrewsPitchfork(percent_change=0.05, fillna=True).update(high, low, close)
# result.median, result.upper, result.lower, result.pivot, result.direction
```

`percent_change` is a fractional threshold (e.g. `0.05` for 5%). Values greater
than `1.0` are treated as percent and divided by 100. The first bar seeds state;
with `fillna=True` early outputs use close as a placeholder for all three lines.

## Theory Of Operation

Alan Andrews' pitchfork draws a median line from an origin pivot \(P_0\) through
the midpoint of the next two pivots \(P_1, P_2\), then places parallel tines
offset by the distance of \(P_1\) from that median. Price often reacts near the
tines.

RTTA does not take manual pivot coordinates. Instead it discovers pivots online
with a percent-change ZigZag on high/low extremes: a high pivot is confirmed
when price reverses down by the threshold from the swing high; a low pivot when
price reverses up from the swing low. The last three confirmed pivots define
the fork; subsequent bars extrapolate the lines using the median slope.

## Recurrence

Let \(H_t, L_t, C_t\) be high, low, close and let \(\tau\) be the percent
threshold. Maintain swing direction \(d_t \in \{-1,0,+1\}\), extreme price and
index, and a ring of up to eight pivots \((p_i, j_i)\) (price and bar index).

**Pivot confirmation (ZigZag).** Starting from the first close, when
\(H_t \ge C_0(1+\tau)\) set direction \(+1\) (upswing); when
\(L_t \le C_0(1-\tau)\) set direction \(-1\). During an upswing, track the
highest high; confirm a high pivot when \(L_t \le E(1-\tau)\). During a
downswing, track the lowest low; confirm a low pivot when \(H_t \ge E(1+\tau)\).
Each confirmation pushes \((E, j_E, \pm 1)\) and flips direction.
`pivot` on the result is \(1\) on confirmation bars and \(0\) otherwise.

**Pitchfork geometry.** When at least three pivots exist, take the last three
as \(P_0, P_1, P_2\) (oldest to newest among those three):

\[
M^{\text{price}} = \tfrac12(p_1 + p_2), \qquad
M^{\text{idx}} = \tfrac12(j_1 + j_2)
\]

\[
s = \frac{M^{\text{price}} - p_0}{M^{\text{idx}} - j_0}
\quad\text{(slope; \(0\) if denominator is zero)}
\]

At the current bar index \(t\):

\[
\begin{aligned}
\operatorname{median}_t &= p_0 + s\,(t - j_0) \\
\delta &= p_1 - \bigl(p_0 + s\,(j_1 - j_0)\bigr) \\
\operatorname{upper}_t &= \operatorname{median}_t + \delta \\
\operatorname{lower}_t &= \operatorname{median}_t - \delta
\end{aligned}
\]

`direction` is the live swing direction \(d_t\). Until three pivots exist,
`fillna=True` returns close for all three lines; `fillna=False` returns `NaN`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class AndrewsPitchfork`. Result fields are `median`, `upper`, `lower`,
`pivot`, and `direction`. The pivot buffer holds at most eight pivots
(oldest dropped).

## Reference

- [ChartSchool: Andrews' Pitchfork](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/andrews-pitchfork)
- [Investopedia: Andrews' Pitchfork](https://www.investopedia.com/terms/a/andrewspitchfork.asp)
