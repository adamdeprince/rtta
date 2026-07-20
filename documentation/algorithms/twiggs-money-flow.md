# TwiggsMoneyFlow

## Summary

`TwiggsMoneyFlow` is RTTA's streaming Twiggs Money Flow. Each bar's volume is
signed by close position within true high/true low, then both the signed flow
and volume are EMA-smoothed; the indicator is their ratio.

## Update API

```python
value = rtta.TwiggsMoneyFlow(window=21, fillna=True).update(high, low, close, volume)
```

With `fillna=False`, output is `NaN` until `window` samples have been seen.

## Theory Of Operation

Colin Twiggs' money flow improves on classic Chaikin Money Flow by using true
high and true low (including the previous close) instead of the raw high-low
range. That reduces gaps' distortion of the close's position in the range. EMA
smoothing of both the accumulation/distribution volume and raw volume replaces
a simple rolling sum, giving a more responsive normalized flow typically in
\((-1, 1)\).

## Recurrence

Let \(H_t, L_t, C_t, V_t\) be high, low, close, volume and \(n\) be `window`
(default \(21\)). On the first bar, true high/low equal \(H_t, L_t\); thereafter:

\[
TH_t = \max(H_t, C_{t-1}), \qquad
TL_t = \min(L_t, C_{t-1})
\]

\[
TR_t = TH_t - TL_t
\]

\[
AD_t =
\begin{cases}
0, & TR_t = 0 \\
\dfrac{(C_t - TL_t) - (TH_t - C_t)}{TR_t}\, V_t, & \text{otherwise}
\end{cases}
\]

\[
\widetilde{AD}_t = \operatorname{EMA}_n(AD_t), \qquad
\widetilde{V}_t = \operatorname{EMA}_n(V_t)
\]

\[
TMF_t = \frac{\widetilde{AD}_t}{\widetilde{V}_t}
\quad\text{(safe divide)}
\]

Both nested EMAs use `fillna=True` and \(\alpha = 2/(n+1)\). Outer warm count is
\(n\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class TwiggsMoneyFlow` with members `ad_` and `vol_` (both `EMA`).

## Reference

- [Incredible Charts: Twiggs Money Flow](https://www.incrediblecharts.com/indicators/twiggs_money_flow.php)
- [ChartSchool: Chaikin Money Flow (related)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-money-flow-cmf)
