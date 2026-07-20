# IntegratedOrderFlowImbalance

## Summary

`IntegratedOrderFlowImbalance` projects multi-level Cont-style OFI onto an online first
principal component of the level-event covariance. Each depth level contributes a rolling
Cont event; an EMA covariance matrix and power-iteration weights form a single integrated
scalar `ofi` plus the level-1 weight `weight_l1`.

## Update API

```python
import rtta

ind = rtta.IntegratedOrderFlowImbalance(
    levels=5, window=1, ema_alpha=0.05, fillna=True
)
result = ind.update(bid_price, bid_size, ask_price, ask_size)  # vectors length `levels`
# result.ofi, result.weight_l1

batch = ind.batch(bid_prices, bid_sizes, ask_prices, ask_sizes)  # (n, levels)
```

`advance(...)` updates state without returning a result. Float32/float64 depth vectors and
batch matrices are supported.

## Theory Of Operation

Raw multi-level OFI is a vector \(v_t \in \mathbb{R}^L\). In practice the components are
highly collinear; Cont-style “integrated” OFI collapses depth into one factor by taking the
leading eigenvector of the event covariance and the corresponding linear combination.
RTTA maintains:

1. Per-level Cont events and rolling sums \(v^{(\ell)}_t\) (same map as multi-level OFI).
2. An exponential moving covariance \(\Sigma_t\) of the vector \(v_t\).
3. Four power-iteration steps on \(\Sigma_t\) starting from the previous weight vector,
   re-oriented so the top-of-book weight is non-negative.
4. Scalar projection \(\mathrm{ofi}_t = w_t^\top v_t\).

Initial weights are \(w = (1,0,\ldots,0)\). With `fillna=False`, outputs are NaN until
`count >= window`.

## Recurrence

Cont events \(e^{(\ell)}_t\) and rolling sums \(v^{(\ell)}_t = S^{(\ell)}_t\) are identical to
`MultiLevelOrderFlowImbalance` (with incomplete windows forced to \(0\) when building \(v\)
under `fillna=True` logic that uses the sum when the window is not full only if `fillna`).

Covariance EMA with \(\alpha = \mathrm{ema\_alpha}\) (clamped to \([10^{-6},1]\)):

\[
\Sigma_t = (1-\alpha)\,\Sigma_{t-1} + \alpha\, v_t v_t^\top.
\]

Power iteration (four steps) from previous weights \(w_{t-1}\):

\[
\tilde{w}^{(0)} = w_{t-1},\qquad
\tilde{w}^{(k+1)} = \frac{\Sigma_t \tilde{w}^{(k)}}{\|\Sigma_t \tilde{w}^{(k)}\|_2},
\quad k=0,1,2,3.
\]

Sign flip so \(\tilde{w}^{(4)}_0 \ge 0\):

\[
w_t =
\begin{cases}
-\tilde{w}^{(4)}, & \tilde{w}^{(4)}_0 < 0,\\
\tilde{w}^{(4)}, & \text{otherwise}.
\end{cases}
\]

Outputs:

\[
\mathrm{ofi}_t = w_t^\top v_t,\qquad
\mathrm{weight\_l1}_t = w_{t,0}.
\]

If `fillna=False` and \(t < W\), both fields are NaN.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class IntegratedOrderFlowImbalance`. Covariance is stored row-major as a length-\(L^2\)
vector. Batch requires shape `(n_samples, levels)`.

## Reference

- [Cont, Kukanov & Stoikov, “The Price Impact of Order Book Events” (arXiv:1011.6402)](https://arxiv.org/abs/1011.6402)
