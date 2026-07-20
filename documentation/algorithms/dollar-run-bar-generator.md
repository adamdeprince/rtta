# DollarRunBarGenerator

## Summary

`DollarRunBarGenerator` is a **dollar-weighted run bar**: while the tick-rule sign holds,
notional volume \(|\mathrm{close}|\cdot\mathrm{volume}\) accumulates; the bar closes when
run dollar volume reaches a threshold.

## Update API

```python
import rtta

ind = rtta.DollarRunBarGenerator(threshold=1.0e6)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`bar_volume` is dollar accumulation of the current run. Flat ticks do not alter the run.
Sign flips restart the run at the current print's dollar volume.

## Theory Of Operation

Dollar run bars combine persistence of order-flow sign with notional-value sampling. They
close when a one-sided streak has moved a fixed amount of money, blending the ideas of
dollar bars and tick run bars from the information-driven bar literature.

## Recurrence

Let \(D^\star > 0\) be `threshold` and \(d_t = |c_t|\,\max(v_t,0)\). Tick sign \(s_t\) is
the same as for volume run bars.

If \(s_t \neq 0\):

\[
\begin{aligned}
s_t = \sigma &\Rightarrow D_{\mathrm{run}} \leftarrow D_{\mathrm{run}} + d_t,\\
s_t \neq \sigma &\Rightarrow \sigma \leftarrow s_t,\ D_{\mathrm{run}} \leftarrow d_t,\ O,H,L \leftarrow c_t.
\end{aligned}
\]

When \(D_{\mathrm{run}} \ge D^\star\):

\[
\begin{aligned}
\mathrm{complete}&=1,\ \mathrm{bars}=1,\ \mathrm{direction}=\sigma,\\
\mathrm{bar\_volume}&=D_{\mathrm{run}},\\
D_{\mathrm{run}}&\leftarrow 0,\ \sigma\leftarrow 0,\ O,H,L\leftarrow c_t.
\end{aligned}
\]

Incomplete ticks report \(\mathrm{bar\_volume}=D_{\mathrm{run}}\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class DollarRunBarGenerator`. Default threshold is \(10^6\).

## Reference

- [López de Prado, *Advances in Financial Machine Learning* (dollar / run bars)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
