# MultiLevelOrderFlowImbalance

## Summary

`MultiLevelOrderFlowImbalance` is RTTA's streaming multi-depth Cont-style order-flow
imbalance. At each limit-order-book level \(\ell = 0,\ldots,L-1\), consecutive best-level
snapshots produce a Cont event contribution; those events are rolled with a fixed-length
sum window, then aggregated into `total`, equal-weight `mean`, and the first five level
series `l1`…`l5`.

## Update API

```python
import numpy as np
import rtta

ind = rtta.MultiLevelOrderFlowImbalance(levels=5, window=1, fillna=True)
# Per tick: depth vectors of length `levels`
result = ind.update(bid_price, bid_size, ask_price, ask_size)
# result.total, result.mean, result.l1 ... result.l5

# Batch: time × depth matrices, shape (n_samples, levels)
batch = ind.batch(bid_prices, bid_sizes, ask_prices, ask_sizes)
```

The `update(...)` call consumes one depth snapshot (four vectors of length `levels`).
`advance(...)` uses the same inputs when the caller wants to update state without
materializing a Python return value. Float32 and float64 arrays are accepted.

## Theory Of Operation

Cont, Kukanov and Stoikov (and later multi-level extensions) define order-flow imbalance
from quote revisions: size added/removed when the bid or ask price steps, or when size
changes at an unchanged price. RTTA applies that event map independently at every book
level, keeps the previous price/size per level, rolls each level's events with
`RollingSumWindow(window)`, and reports:

- `total` — sum of rolled events over all levels
- `mean` — `total / levels`
- `l1`…`l5` — rolled contribution of the first five levels (zeros if fewer levels exist)

The first snapshot contributes zeros (no previous quote). With `fillna=False`, incomplete
rolling windows emit NaN for that level and are omitted from `total`.

## Recurrence

Let \(L\) be `levels`, \(W\) be `window`, and at time \(t\) let
\((b^{(\ell)}_t, B^{(\ell)}_t, a^{(\ell)}_t, A^{(\ell)}_t)\) be bid price, bid size, ask
price, and ask size at level \(\ell\).

The Cont event at level \(\ell\) (after the first observation) is

\[
e^{(\ell)}_t
=
\mathbf{1}\{b^{(\ell)}_t \ge b^{(\ell)}_{t-1}\} B^{(\ell)}_t
-
\mathbf{1}\{b^{(\ell)}_t \le b^{(\ell)}_{t-1}\} B^{(\ell)}_{t-1}
-
\mathbf{1}\{a^{(\ell)}_t \le a^{(\ell)}_{t-1}\} A^{(\ell)}_t
+
\mathbf{1}\{a^{(\ell)}_t \ge a^{(\ell)}_{t-1}\} A^{(\ell)}_{t-1}.
\]

On the first tick, \(e^{(\ell)}_t = 0\). Let \(S^{(\ell)}_t\) be the rolling sum of the last
\(\min(t,W)\) events at level \(\ell\):

\[
S^{(\ell)}_t = \sum_{k=0}^{\min(t,W)-1} e^{(\ell)}_{t-k}
\quad\text{(or NaN if `fillna=False` and the window is not yet full).}
\]

Aggregates:

\[
\text{total}_t = \sum_{\ell=0}^{L-1} S^{(\ell)}_t,\qquad
\text{mean}_t = \frac{\text{total}_t}{L},\qquad
\mathrm{l}_{j,t} = S^{(j-1)}_t \quad (j=1,\ldots,5).
\]

State after each tick stores the current \((b,B,a,A)\) as previous for the next event.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class MultiLevelOrderFlowImbalance` (helper `cont_ofi_event`). Batch paths require
C-contiguous matrices of shape `(n_samples, levels)`.

## Reference

- [Cont, Kukanov & Stoikov, “The Price Impact of Order Book Events” (arXiv:1011.6402)](https://arxiv.org/abs/1011.6402)
