# IntradayIntensity

## Summary

`IntradayIntensity` is RTTA's streaming windowed average of volume-weighted
intraday intensity. Each bar contributes an intensity flow
\(((2C - H - L)/(H - L)) \cdot V\); the indicator returns the ratio of the
rolling sum of those flows to the rolling sum of volume.

## Update API

```python
value = rtta.IntradayIntensity(window=21, fillna=True).update(high, low, close, volume)
```

With `fillna=False`, output is `NaN` until the window buffer is full.

## Theory Of Operation

David Bostian's Intraday Intensity (also related to the Accumulation/Distribution
and money-flow family) locates the close within the bar's high-low range and
weights that position by volume. Closes near the high produce positive intensity;
closes near the low produce negative intensity. Averaging flow over volume across
a rolling window yields a normalized participation measure in roughly \([-1, 1]\).

## Recurrence

Let \(H_t, L_t, C_t, V_t\) be high, low, close, volume and \(n\) be `window`
(default \(21\)).

\[
II^{\text{raw}}_t =
\begin{cases}
0, & H_t = L_t \\
\dfrac{2C_t - H_t - L_t}{H_t - L_t}\, V_t, & \text{otherwise}
\end{cases}
\]

Maintain rolling sums over the last \(\min(t,n)\) samples (fixed-capacity buffer
with FIFO eviction when full):

\[
F_t = \sum_{i \in W_t} II^{\text{raw}}_i, \qquad
U_t = \sum_{i \in W_t} V_i
\]

\[
\operatorname{IntradayIntensity}_t = \frac{F_t}{U_t}
\quad\text{(safe divide)}
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class IntradayIntensity` using two rolling sum buffers (`flow_`, `vol_`).

## Reference

- [Investopedia: Intraday Intensity Index](https://www.investopedia.com/terms/i/intradayintensityindex.asp)
- [ChartSchool: Accumulation Distribution Line](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/accumulation-distribution-line)
