# ResidualFOCuS

## Summary

`ResidualFOCuS` applies the FOCuS mean changepoint detector to a **residual or
innovation** series (model errors, hedge residuals, filter innovations, etc.). The
engine is identical to `FOCuS`; the class name documents the intended input semantics.

## Update API

```python
import rtta

ind = rtta.ResidualFOCuS(threshold=10.0, mu0=0.0, sigma=1.0, max_candidates=200)
result = ind.update(residual)
# result.signal ∈ {-1, 0, +1}, result.statistic
```

`advance(...)` updates state without returning a result. Constructor parameters match
`FOCuS`.

## Theory Of Operation

Model-based monitoring often reduces to “is the residual still zero-mean noise?” Feeding
pre-whitened or model residuals into FOCuS detects mean shifts that pure price-level
CUSUM would confuse with trend. Typical pipelines:

1. Fit a predictive model online (regression, Kalman, pairs residual).
2. Stream \(r_t = y_t - \hat{y}_t\) (or a z-scored innovation) into `ResidualFOCuS`.
3. Fire when the residual mean changes beyond the GLR threshold.

Under a correctly specified model, \(\mu_0 = 0\) and \(\sigma\) should match residual scale.

## Recurrence

Identical to `FOCuS` with observation \(x_t\) replaced by residual \(r_t\):

\[
y_t = r_t - \mu_0,
\]

candidates \((S,n)\) updated as \((S+y_t, n+1)\) plus a new \((y_t,1)\), pruned by mean
dominance, and

\[
\Lambda_t = \max \frac{S^2}{2\sigma^2 n},\qquad
\mathrm{signal}_t = \operatorname{sign}(S^\star)\ \text{if}\ \Lambda_t \ge h\ \text{else}\ 0.
\]

See [focus.md](focus.md) for the full FOCuS recurrence and pruning details.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ResidualFOCuS` as a
member `FOCuS focus_`; `update`/`advance`/`batch_array`/`last` forward to that engine.

## Reference

- [Romano et al., FOCuS (arXiv:2110.08205)](https://arxiv.org/abs/2110.08205)
