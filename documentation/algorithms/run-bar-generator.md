# RunBarGenerator

## Summary

`RunBarGenerator` implements López de Prado **tick run bars**: a bar closes when the count
of consecutive same-sign ticks (tick rule on close) reaches an integer threshold.
Optional `update(close, volume)` still completes on tick-count but reports accumulated
run volume as `bar_volume`.

## Update API

```python
import rtta

ind = rtta.RunBarGenerator(threshold=10)
result = ind.update(close)                 # tick-count runs
result = ind.update(close, volume)         # same rule; bar_volume = run volume
# result.bar_open, bar_close, bar_high, bar_low, bar_volume,
# result.direction, result.complete, result.bars
```

Flat ticks (\(c_t = c_{t-1}\)) do not break the run and do not increment the count.
An opposite sign starts a new run at count 1 with OHLC reset to the current close.

## Theory Of Operation

Run bars sample when a streak of buyer- or seller-initiated prints ends by length, not
by calendar time. They emphasize persistence of order flow: long same-sign sequences
close bars more often. RTTA's primary definition is **tick count**; the two-argument
overload accumulates volume for reporting but still thresholds on `run_count`.

## Recurrence

Let \(N^\star = \max(\mathrm{threshold}, 1)\). Tick sign:

\[
s_t =
\begin{cases}
+1, & c_t > c_{t-1},\\
-1, & c_t < c_{t-1},\\
0, & \text{flat (ignored for run logic)}.
\end{cases}
\]

If \(s_t \neq 0\):

\[
\begin{aligned}
s_t = \sigma \text{ (current run sign)} &\Rightarrow n \leftarrow n+1,\\
s_t \neq \sigma &\Rightarrow \sigma \leftarrow s_t,\ n \leftarrow 1,\ O,H,L \leftarrow c_t.
\end{aligned}
\]

When \(n \ge N^\star\):

\[
\mathrm{complete}=1,\ \mathrm{bars}=1,\ \mathrm{direction}=\sigma,\
\mathrm{bar\_volume}=n\ \text{(or run volume if volume overload)},\
n\leftarrow 0,\ \sigma\leftarrow 0,\ O,H,L\leftarrow c_t.
\]

Volume overload: same sign rule; \(V_{\mathrm{run}} \mathrel{+}= v_t^+\) on continuation;
completion still requires \(n \ge N^\star\), then \(V_{\mathrm{run}}\leftarrow 0\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RunBarGenerator`.
Default threshold is 10 ticks.

## Reference

- [López de Prado, *Advances in Financial Machine Learning* (run bars)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
