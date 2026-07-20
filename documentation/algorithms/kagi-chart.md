# KagiChart

## Summary

`KagiChart` is a streaming Kagi line: the line tracks price in the current direction and
**reverses** only after an absolute move of size `reversal` against the line. Outputs are
the current line level, yang/yin direction, and a reversal flag.

## Update API

```python
import rtta

ind = rtta.KagiChart(reversal=1.0)
result = ind.update(price)
# result.line, result.direction (+1 yang / -1 yin / 0 unset),
# result.reversal (1 if direction flipped this tick else 0)
```

`reversal` is an absolute price amount (not percent), floored at \(10^{-12}\).

## Theory Of Operation

Kagi charts ignore time and small noise: the line extends as long as price continues in
the active direction; only a counter-move of at least the reversal amount flips yang
(up, \(+1\)) to yin (down, \(-1\)) or vice versa. Classic charting also uses thick/thin
lines at prior highs/lows; this implementation exposes the line, signed direction, and
reversal events for quantitative use.

## Recurrence

Let \(\delta =\)`reversal`. First price: \(\ell = p_0\), direction \(d=0\),
\(\mathrm{reversal}=0\).

If \(d \ge 0\) (flat or yang):

\[
\begin{aligned}
p_t > \ell &\Rightarrow \ell \leftarrow p_t
\quad\text{(and if \(d=0\), set \(d\leftarrow +1\))},\\
p_t \le \ell - \delta &\Rightarrow d \leftarrow -1,\ \ell \leftarrow p_t,\
\mathrm{reversal}\leftarrow 1.
\end{aligned}
\]

If \(d < 0\) (yin):

\[
\begin{aligned}
p_t < \ell &\Rightarrow \ell \leftarrow p_t,\\
p_t \ge \ell + \delta &\Rightarrow d \leftarrow +1,\ \ell \leftarrow p_t,\
\mathrm{reversal}\leftarrow 1.
\end{aligned}
\]

Outputs: \(\mathrm{line}_t=\ell\), \(\mathrm{direction}_t=d\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class KagiChart`.
Result type is `KagiChartResult`.

## Reference

- [Investopedia — Kagi Chart](https://www.investopedia.com/terms/k/kagichart.asp)
