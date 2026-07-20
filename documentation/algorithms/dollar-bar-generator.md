# DollarBarGenerator

## Summary

`DollarBarGenerator` closes information bars when cumulative **dollar volume**
\(|\mathrm{close}|\cdot\mathrm{volume}\) reaches a threshold. It is the notional-value
counterpart of volume bars in the López de Prado information-bar family.

## Update API

```python
import rtta

ind = rtta.DollarBarGenerator(threshold=1.0e6)
result = ind.update(close, volume)
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

`bar_volume` is dollar accumulation (threshold units), not share count. `complete` /
`bars` / `direction` follow the same conventions as `VolumeBarGenerator`.

## Theory Of Operation

Dollar (notional) bars sample the market whenever a fixed amount of money changes hands.
This partially normalizes for price level: a \$1M threshold means roughly the same
economic activity whether the instrument trades at \$10 or \$1000. RTTA accumulates
\(|c_t|\cdot v_t^+\) and splits overshoot across multiple bars within one update.

## Recurrence

Let \(D^\star > 0\) be `threshold` and \(d_t = |c_t|\,\max(v_t,0)\).

Accumulate \(A_t \leftarrow A_{t-1} + d_t\) (after updating running high/low of \(c_t\)).
While \(A_t \ge D^\star\):

\[
\begin{aligned}
\mathrm{complete} &\leftarrow 1,\quad
\mathrm{bars} \mathrel{+}= 1,\\
\mathrm{direction} &\leftarrow
\begin{cases}+1,& c_t \ge O\\ -1,& c_t < O\end{cases},\\
\text{emit} &\ (O,c_t,H,L),\ \mathrm{bar\_volume}=D^\star,\\
A_t &\leftarrow A_t - D^\star,\quad O,H,L \leftarrow c_t.
\end{aligned}
\]

Incomplete ticks report \(\mathrm{bar\_volume}=A_t\). First-tick initialization seeds
OHLC at \(c_t\) and may complete immediately if \(d_t \ge D^\star\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class DollarBarGenerator`.
Default threshold is \(10^6\).

## Reference

- [López de Prado, *Advances in Financial Machine Learning* (dollar bars)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
