# WeightedMultiPeerOrderFlowImbalance

## Summary

`WeightedMultiPeerOrderFlowImbalance` is RTTA's streaming basket peer OFI with
explicit peer weights: a weighted mean of peer OFIs each tick, then a rolling
beta of own return on that peer mean (impact and residual).

## Update API

```python
# peer_ofis and weights shape (n_peers,)
result = rtta.WeightedMultiPeerOrderFlowImbalance(window=50).update(
    own_return, peer_ofis, weights
)
# result.beta, result.impact, result.residual, result.peer_mean

# batch: own_return (T,), peer_ofis (T, P), weights (T, P)
batch = ind.batch(own_return, peer_ofis, weights)
```

Equal weights recover the peer mean of `MultiPeerOrderFlowImbalance`.
Non-positive or non-finite weights are skipped; if every weight is invalid, the
implementation falls back to equal weighting.

## Theory Of Operation

Equal-weight basket OFI treats every peer name the same. Weighted multi-peer OFI
lets the caller supply liquidity, ADV, beta, or sector weights so the peer
pressure index is a portfolio, not a simple average. Rolling OLS beta of own
return on that weighted peer mean produces Cont-style cross-asset impact and a
residual idiosyncratic return.

## Recurrence

Let \(r_t\) be own return, \(f_{t,i}\) peer OFIs, and \(w_{t,i}\) peer weights
for \(i=1,\ldots,P\). Let \(W_t = \sum_i w_{t,i}\) over positive finite weights.

\[
\bar f_t = \frac{\sum_i w_{t,i} f_{t,i}}{W_t}
\]

Maintain rolling pair statistics over the last \(n\) samples (`window`) of
\((r_t, \bar f_t)\) and form OLS beta \(\hat\beta_t\). Then

\[
\begin{aligned}
\operatorname{impact}_t &= \hat\beta_t\,\bar f_t \\
\operatorname{residual}_t &= r_t - \operatorname{impact}_t \\
\operatorname{peer\_mean}_t &= \bar f_t
\end{aligned}
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class WeightedMultiPeerOrderFlowImbalance` using `RollingPairStats`. Unlike
equal-weight `MultiPeerOrderFlowImbalance`, every update requires a weight
vector (or weight matrix in batch).

## Reference

- [Cont, Kukanov, and Stoikov, "The Price Impact of Order Book Events,"
  arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- [Cross-asset / multi-name flow pressure context in market
  microstructure surveys](https://arxiv.org/abs/1011.6402)
