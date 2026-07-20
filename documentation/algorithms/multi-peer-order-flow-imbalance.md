# MultiPeerOrderFlowImbalance

## Summary

`MultiPeerOrderFlowImbalance` extends cross-asset OFI impact to a **basket of peers**.
Each tick, peer OFI values are averaged with equal weights into `peer_mean`; rolling OLS
of own return on that mean yields `beta`, `impact`, and `residual`.

## Update API

```python
import numpy as np
import rtta

ind = rtta.MultiPeerOrderFlowImbalance(window=50, fillna=True)
result = ind.update(own_return, peer_ofis)  # peer_ofis shape (n_peers,)
# result.beta, result.impact, result.residual, result.peer_mean

# Batch: own_return (T,), peer_ofis (T, n_peers)
batch = ind.batch(own_returns, peer_ofi_matrix)
```

`advance(...)` updates state without returning a result. Empty peer vectors yield NaNs.

## Theory Of Operation

Multi-name flow often arrives as a vector of peer OFIs. An equal-weight reduction

\[
\bar{f}_t = \frac{1}{P}\sum_{p=1}^{P} f_{t,p}
\]

acts as a single cross-impact regressor. Rolling \(\beta\) of own return on \(\bar{f}\)
matches `CrossAssetOrderFlowImbalance` with peer feature \(\bar{f}_t\). This is a lightweight
basket alternative to weighted multi-peer variants.

## Recurrence

For peer vector \(f_t \in \mathbb{R}^P\) (\(P \ge 1\)):

\[
\bar{f}_t = \frac{1}{P}\sum_{p=1}^{P} f_{t,p}.
\]

Push \((x_t, y_t) = (r_t, \bar{f}_t)\) into `RollingPairStats` of length \(W\). With
rolling sums \(S_x,S_y,S_{xy},S_{y^2}\) and \(n = |W_t|\):

\[
\beta_t = \frac{n\,S_{xy} - S_x S_y}{n\,S_{y^2} - S_y^2},
\]

\[
\mathrm{impact}_t = \beta_t\,\bar{f}_t,\qquad
\mathrm{residual}_t = r_t - \mathrm{impact}_t,\qquad
\mathrm{peer\_mean}_t = \bar{f}_t.
\]

If `fillna=False` and the window is incomplete, \(\beta\), impact, and residual are NaN;
`peer_mean` is still \(\bar{f}_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class MultiPeerOrderFlowImbalance`. Batch requires at least one peer column. Float32 and
float64 peer vectors are supported.

## Reference

- [Cont, Kukanov & Stoikov, “The Price Impact of Order Book Events” (arXiv:1011.6402)](https://arxiv.org/abs/1011.6402)
