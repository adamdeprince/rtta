# RSI

## Summary

`RSI` computes RTTA's incremental Relative Strength Index over one scalar price
stream. It stores the previous value plus running positive and negative move
state.

## Update API

```python
value = rtta.RSI(window=14, fillna=True).update(value)
```

The first sample initializes `prev`. With `fillna=True`, the first returned
value is `50.0`; with `fillna=False`, warmup returns `NaN`.

## Theory Of Operation

RSI compares recent upward movement against recent downward movement and maps
that ratio into a bounded oscillator. The RTTA C++ implementation updates the
upward state on positive moves and the downward state on negative moves. During
warmup it uses the current sample count as the averaging denominator; after
warmup it uses Wilder-style smoothing with `window`.

## Recurrence

Let \(x_t\) be the current value, \(x_{t-1}\) the prior value, and \(n\) be
`window`.

\[
g_t = \max(x_t - x_{t-1}, 0), \qquad
\ell_t = \max(x_{t-1} - x_t, 0)
\]

During warmup, RTTA updates only the side that moved:

\[
G_t =
\begin{cases}
\frac{(t-1)G_{t-1}+g_t}{t}, & g_t > 0 \\
G_{t-1}, & g_t = 0
\end{cases}
\]

\[
L_t =
\begin{cases}
\frac{(t-1)L_{t-1}+\ell_t}{t}, & \ell_t > 0 \\
L_{t-1}, & \ell_t = 0
\end{cases}
\]

After warmup, the same directional update uses \(n\):

\[
G_t =
\begin{cases}
\frac{(n-1)G_{t-1}+g_t}{n}, & g_t > 0 \\
G_{t-1}, & g_t = 0
\end{cases}
\qquad
L_t =
\begin{cases}
\frac{(n-1)L_{t-1}+\ell_t}{n}, & \ell_t > 0 \\
L_{t-1}, & \ell_t = 0
\end{cases}
\]

The output is:

\[
RS_t = \frac{G_t/n}{L_t/n}, \qquad
RSI_t = 100 - \frac{100}{1 + RS_t}
\]

If the downside state is zero, RTTA returns `100.0` once it is past the initial
sample path.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RSI`. The directional
state-update behavior above follows the current C++ code exactly.

## Reference

- [ChartSchool: Relative Strength Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi)
