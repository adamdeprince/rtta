# KalmanInnovationResidualBOCPD

## Summary

`KalmanInnovationResidualBOCPD` chains a **Kalman innovation z-score** residual into
**ResidualBOCPD** (bounded BOCPD). It detects Bayesian online changepoints in the filter
innovation process and returns signal, change probability (`score`), and residual.

## Update API

```python
import rtta

ind = rtta.KalmanInnovationResidualBOCPD(
    max_run_length=128,
    hazard=0.01,
    threshold=0.5,
    min_variance=1e-6,
    initial_price=float("nan"),
    dt=1.0,
    measurement_variance=0.25,
    fillna=True,
)
result = ind.update(close)
# result.signal, result.score (= probability), result.residual
```

Non-finite innovations skip BOCPD and return zero signal/score with the residual echoed.

## Theory Of Operation

Same residual construction as `KalmanInnovationResidualFOCuS`: a constant-velocity Kalman
filter produces \(\rho_t = \nu_t / \sqrt{S_t}\). Instead of FOCuS GLR, **BOCPD** places a
posterior on residual run length and flags a changepoint when \(P(r_t=0) \ge \tau\). This
is complementary to FOCuS: BOCPD models hazard and online residual mean/variance per
hypothesis rather than a fixed Gaussian CUSUM threshold.

## Recurrence

**1. Innovation z-score** \(\rho_t\) as in `KalmanInnovationZScore` / residual-FOCuS docs:

\[
\nu_t = c_t - H\hat{x}_{t|t-1},\qquad
\rho_t = \nu_t / \sqrt{H P_{t|t-1} H^\top + R}.
\]

**2. ResidualBOCPD** on \(\rho_t\) with parameters
`(max_run_length, hazard, threshold, min_variance)` — full recurrence in
[residual-bocpd.md](residual-bocpd.md) / [bounded-bocpd.md](bounded-bocpd.md):

\[
\mathrm{score}_t = \pi_t(0),\qquad
\mathrm{signal}_t = \mathbf{1}\{\pi_t(0) \ge \tau\},\qquad
\mathrm{residual}_t = \rho_t.
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class KalmanInnovationResidualBOCPD` (`KalmanInnovationZScore innov_` +
`ResidualBOCPD bocpd_`). Result type is `InnovationChangepointResult` (`score` holds BOCPD
probability).

## Reference

- [Adams & MacKay, “Bayesian Online Changepoint Detection” (arXiv:0710.3742)](https://arxiv.org/abs/0710.3742)
- [Welch & Bishop, “An Introduction to the Kalman Filter”](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
