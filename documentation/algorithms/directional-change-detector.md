# DirectionalChangeDetector

## Summary

`DirectionalChangeDetector` samples **intrinsic time** via directional-change (DC)
events: a move of relative size \(\theta\) from the last extremum defines a DC; the
path beyond that event is reported as overshoot. Outputs include event flag, overshoot,
current extremum, and trend mode.

## Update API

```python
import rtta

ind = rtta.DirectionalChangeDetector(threshold=0.01)  # 1% relative
result = ind.update(price)
# result.event ∈ {-1, 0, +1},
# result.overshoot, result.extremum, result.direction
```

`threshold` is a relative fraction (\(0.01 = 1\%\)). `direction` / mode is \(+1\) in an
uptrend (seeking a downturn), \(-1\) in a downtrend (seeking an upturn), \(0\) until the
first DC.

## Theory Of Operation

Directional-change methods (Glattfelder, Dupuis, Olsen and related intrinsic-time work)
replace calendar sampling with event sampling: time advances when price has moved by a
fixed relative amount from a local extreme. After a DC up, the algorithm tracks the new
high as extremum until a \(\theta\) drop; after a DC down, it tracks lows until a
\(\theta\) rise. Overshoot measures how far price has continued beyond the last DC
price in the current mode.

## Recurrence

Let \(\theta =\)`threshold` \(> 0\). On the first price \(p_0\):
extremum \(E = p_0\), last DC price \(p^{\mathrm{dc}} = p_0\), mode \(m=0\), event \(0\).

**Bootstrap** (\(m=0\)):

\[
\begin{aligned}
p_t \ge E(1+\theta) &\Rightarrow m\leftarrow +1,\ \mathrm{event}\leftarrow +1,\
p^{\mathrm{dc}},E\leftarrow p_t,\\
p_t \le E(1-\theta) &\Rightarrow m\leftarrow -1,\ \mathrm{event}\leftarrow -1,\
p^{\mathrm{dc}},E\leftarrow p_t,\\
\text{else} &\Rightarrow E\leftarrow \mathrm{clip\ extend}(E,p_t).
\end{aligned}
\]

**Uptrend** (\(m=+1\)): \(E \leftarrow \max(E, p_t)\). If
\(p_t \le E(1-\theta)\), set \(m\leftarrow -1\), \(\mathrm{event}\leftarrow -1\),
\(p^{\mathrm{dc}},E\leftarrow p_t\).

**Downtrend** (\(m=-1\)): \(E \leftarrow \min(E, p_t)\). If
\(p_t \ge E(1+\theta)\), set \(m\leftarrow +1\), \(\mathrm{event}\leftarrow +1\),
\(p^{\mathrm{dc}},E\leftarrow p_t\).

Overshoot (when \(p^{\mathrm{dc}} > 0\)):

\[
\mathrm{overshoot}_t =
\begin{cases}
(p_t - p^{\mathrm{dc}})/p^{\mathrm{dc}}, & m=+1,\\
(p^{\mathrm{dc}} - p_t)/p^{\mathrm{dc}}, & m=-1,\\
0, & m=0.
\end{cases}
\]

Outputs: \(\mathrm{extremum}_t = E\), \(\mathrm{direction}_t = m\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class DirectionalChangeDetector`. Result type is `DirectionalChangeResult`.

## Reference

- [Glattfelder, Dupuis & Olsen, “Patterns in high-frequency FX data: discovery of 12 empirical scaling laws” (arXiv:0809.1040)](https://arxiv.org/abs/0809.1040)
