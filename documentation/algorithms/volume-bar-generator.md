# VolumeBarGenerator

## Summary

`VolumeBarGenerator` is a Lopez de Prado–style **information bar** that closes a bar when
cumulative trade volume reaches a fixed threshold. Each `update(close, volume)` advances
the in-progress OHLC and may complete one or more bars if volume overshoots.

## Update API

```python
import rtta

ind = rtta.VolumeBarGenerator(threshold=10000.0)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`complete` is \(1\) when at least one bar closed on this tick; `bars` is the count of bars
closed this tick (can exceed 1 if a single print carries multiple thresholds of volume).
`direction` is \(+1\) if the completed bar's close \(\ge\) open, else \(-1\) (0 if none
completed). Incomplete ticks report running `bar_volume = accum`.

## Theory Of Operation

Clock time is irregular; volume time normalizes activity so each bar embeds roughly the
same amount of traded size. RTTA accumulates \(\max(\mathrm{volume}, 0)\) and, while
accumulation stays at or above the threshold, emits a completed bar of volume equal to
`threshold`, subtracts that amount, and restarts OHLC at the current close (so large
prints can spawn multiple bars in one call).

## Recurrence

Let \(V^\star > 0\) be `threshold`. On the first tick, set
\(O=H=L=c_t\), \(A_t = v_t^+\), and if \(A_t \ge V^\star\) complete a bar and reset.

Thereafter:

\[
H \leftarrow \max(H, c_t),\quad L \leftarrow \min(L, c_t),\quad
A_t \leftarrow A_{t-1} + v_t^+,\quad v_t^+ = \max(v_t, 0).
\]

While \(A_t \ge V^\star\):

\[
\begin{aligned}
\mathrm{complete} &\leftarrow 1,\\
\mathrm{bars} &\leftarrow \mathrm{bars}+1,\\
\mathrm{direction} &\leftarrow \mathbf{1}\{c_t \ge O\} - \mathbf{1}\{c_t < O\},\\
\text{emit OHLC} &\leftarrow (O, c_t, H, L),\quad
\mathrm{bar\_volume} \leftarrow V^\star,\\
A_t &\leftarrow A_t - V^\star,\quad
O,H,L \leftarrow c_t.
\end{aligned}
\]

If no bar completed, reported volume is the running accumulation \(A_t\); otherwise the
last completed bar's fields (volume \(V^\star\)) are returned for this tick.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolumeBarGenerator`.
Result type is `InformationBarResult`.

## Reference

- [López de Prado, *Advances in Financial Machine Learning* (information-driven bars)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
