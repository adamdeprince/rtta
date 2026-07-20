# VolumeRunBarGenerator

## Summary

`VolumeRunBarGenerator` is a **volume-weighted run bar**: while the tick-rule sign stays
the same, trade volume accumulates; the bar closes when run volume reaches a threshold
(not tick count).

## Update API

```python
import rtta

ind = rtta.VolumeRunBarGenerator(threshold=10000.0)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

Flat ticks leave the run unchanged. An opposite sign starts a new run with volume equal
to the current print and OHLC reset to that close.

## Theory Of Operation

Standard run bars count ticks; volume run bars weight each same-sign tick by its size so
that a few large prints can complete a bar as fast as many small ones. This is the run
analogue of volume bars within a persistent-sign regime.

## Recurrence

Let \(V^\star > 0\) be `threshold` and

\[
s_t =
\begin{cases}
+1, & c_t > c_{t-1},\\
-1, & c_t < c_{t-1},\\
0, & \text{flat}.
\end{cases}
\]

If \(s_t \neq 0\):

\[
\begin{aligned}
s_t = \sigma &\Rightarrow V_{\mathrm{run}} \leftarrow V_{\mathrm{run}} + v_t^+,\\
s_t \neq \sigma &\Rightarrow \sigma \leftarrow s_t,\ V_{\mathrm{run}} \leftarrow v_t^+,\ O,H,L \leftarrow c_t,
\end{aligned}
\]

where \(v_t^+ = \max(v_t,0)\). High/low track \(c_t\) while the run is active.

When \(V_{\mathrm{run}} \ge V^\star\):

\[
\begin{aligned}
\mathrm{complete}&=1,\ \mathrm{bars}=1,\ \mathrm{direction}=\sigma,\\
\mathrm{bar\_volume}&=V_{\mathrm{run}},\\
V_{\mathrm{run}}&\leftarrow 0,\ \sigma\leftarrow 0,\ O,H,L\leftarrow c_t.
\end{aligned}
\]

Otherwise \(\mathrm{bar\_volume}=V_{\mathrm{run}}\), `complete`/`bars`/`direction` zero
(except incomplete runs still report direction only on completion).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class VolumeRunBarGenerator`. Result type is `InformationBarResult`.

## Reference

- [López de Prado, *Advances in Financial Machine Learning* (run bars / volume sampling)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
