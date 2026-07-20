# ImbalanceBarGenerator

## Summary

`ImbalanceBarGenerator` builds López de Prado–style **volume imbalance bars**: signed
volume (tick-rule sign on close) accumulates until its absolute value hits a threshold,
then the bar closes and the imbalance resets.

## Update API

```python
import rtta

ind = rtta.ImbalanceBarGenerator(threshold=10000.0)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`bar_volume` is \(|I_t|\) (absolute cumulative signed volume). `direction` on completion
is the sign of the imbalance. Flat ticks (unchanged close) contribute sign \(0\).

## Theory Of Operation

Imbalance bars close when buy and sell volume become sufficiently asymmetric, so bar
frequency rises in one-sided markets and falls when flow is balanced. Sign uses the
tick rule: uptick \(\Rightarrow +1\), downtick \(\Rightarrow -1\), flat \(\Rightarrow 0\).
Unlike volume/dollar bars, overshoot is not split into multiple bars in one tick: a
single completion resets imbalance to zero.

## Recurrence

Let \(V^\star > 0\) be `threshold`. After the first tick (seed OHLC, \(I=0\)):

\[
s_t =
\begin{cases}
+1, & c_t > c_{t-1},\\
-1, & c_t < c_{t-1},\\
0, & c_t = c_{t-1},
\end{cases}
\qquad
I_t = I_{t-1} + s_t\,\max(v_t,0).
\]

Update running \(H,L\) from \(c_t\). If \(|I_t| \ge V^\star\):

\[
\begin{aligned}
\mathrm{complete}&=1,\ \mathrm{bars}=1,\\
\mathrm{direction}&=\operatorname{sign}(I_t)\quad(+1\text{ if }I_t\ge 0),\\
\mathrm{bar\_volume}&=|I_t|,\\
I &\leftarrow 0,\quad O,H,L \leftarrow c_t.
\end{aligned}
\]

Otherwise \(\mathrm{complete}=0\), \(\mathrm{bar\_volume}=|I_t|\), direction \(0\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class ImbalanceBarGenerator`. Result type is `InformationBarResult`.

## Reference

- [López de Prado, *Advances in Financial Machine Learning* (imbalance bars)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
