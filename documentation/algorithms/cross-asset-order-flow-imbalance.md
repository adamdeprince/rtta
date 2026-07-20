# CrossAssetOrderFlowImbalance

## Summary

`CrossAssetOrderFlowImbalance` estimates the rolling linear impact of a **peer** order-flow
imbalance series on the **own** asset return: \(\beta\), fitted impact \(\beta\cdot\mathrm{peer\_ofi}\),
and residual return. It is a Cont-style cross-impact feature for multi-name flow.

## Update API

```python
import rtta

ind = rtta.CrossAssetOrderFlowImbalance(window=50, fillna=True)
result = ind.update(own_return, peer_ofi)
# result.beta, result.impact, result.residual, result.peer_ofi
```

`advance(...)` updates state without returning a result. Batch accepts aligned
`own_return` and `peer_ofi` arrays.

## Theory Of Operation

Given pairs \((r_t, f_t)\) where \(r_t\) is the own return and \(f_t\) is peer OFI (or any
peer flow feature), the class maintains rolling sums of \(r\), \(f\), \(r^2\), \(f^2\),
and \(rf\) over a window of length \(W\). Beta is the OLS slope of \(r\) on \(f\)
(no intercept term in the moment form used by `RollingPairStats::beta`):

\[
\beta = \frac{\mathrm{Cov}(r,f)}{\mathrm{Var}(f)}.
\]

Cross-impact and residual then follow the linear model \(r \approx \beta f\).

## Recurrence

After each push of \((x_t, y_t) = (r_t, f_t)\) into a window of capacity \(W\), with
running sums \(S_x, S_y, S_{x^2}, S_{y^2}, S_{xy}\) and \(n = |W_t|\):

\[
\beta_t = \frac{n\,S_{xy} - S_x S_y}{n\,S_{y^2} - S_y^2}
\quad\text{(safe divide; \(0\) if variance is zero)}.
\]

\[
\mathrm{impact}_t = \beta_t\, f_t,\qquad
\mathrm{residual}_t = r_t - \mathrm{impact}_t,\qquad
\mathrm{peer\_ofi}_t = f_t.
\]

If `fillna=False` and the window is not full, \(\beta\), impact, and residual are NaN
while `peer_ofi` still echoes \(f_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class CrossAssetOrderFlowImbalance` using `RollingPairStats` (`x = own_return`,
`y = peer_ofi`). Window length is at least 2.

## Reference

- [Cont, Kukanov & Stoikov, “The Price Impact of Order Book Events” (arXiv:1011.6402)](https://arxiv.org/abs/1011.6402)
