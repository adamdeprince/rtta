# KalmanInnovationResidualFOCuS

## Summary

`KalmanInnovationResidualFOCuS` chains a **Kalman innovation z-score** residual into
**FOCuS** changepoint detection. Each price update produces a normalized innovation; FOCuS
monitors that residual for mean shifts and returns signal, FOCuS statistic (`score`), and
the residual itself.

## Update API

```python
import rtta

ind = rtta.KalmanInnovationResidualFOCuS(
    focus_threshold=10.0,
    focus_sigma=1.0,
    initial_price=float("nan"),
    dt=1.0,
    measurement_variance=0.25,
    fillna=True,
)
result = ind.update(close)
# result.signal, result.score, result.residual
```

If the innovation is non-finite, FOCuS is not advanced and `signal`/`score` stay 0 with
the non-finite residual echoed.

## Theory Of Operation

Price levels are nonstationary; innovating a constant-velocity Kalman filter yields
approximately white residuals under the model. Scaling by the predicted innovation
standard deviation produces a z-score residual suitable for zero-mean FOCuS
(\(\mu_0 = 0\)). Changepoints then flag structural breaks: volatility jumps that
mis-scale the innovation, velocity regime changes, or measurement anomalies that FOCuS
sees as residual mean shifts.

Internal Kalman defaults match `KalmanInnovationZScore` process/measurement settings
used by the convenience constructor (position/velocity process variances \(10^{-4}\) /
\(10^{-3}\), unit initial variances, etc.).

## Recurrence

**1. Innovation z-score** (constant-velocity Kalman, measurement \(z_t = c_t\)):

\[
\hat{x}_{t|t-1} = F\hat{x}_{t-1|t-1},\quad
P_{t|t-1} = F P_{t-1|t-1} F^\top + Q,
\]

\[
\nu_t = z_t - H\hat{x}_{t|t-1},\quad
S_t = H P_{t|t-1} H^\top + R,\quad
\rho_t = \frac{\nu_t}{\sqrt{S_t}}.
\]

**2. FOCuS on \(\rho_t\)** with threshold \(h=\)`focus_threshold`, \(\mu_0=0\),
\(\sigma=\)`focus_sigma` (see [focus.md](focus.md)):

\[
y_t = \rho_t,\quad
\Lambda_t = \max_{(S,n)} \frac{S^2}{2\sigma^2 n},\quad
\mathrm{signal}_t =
\begin{cases}
\pm 1, & \Lambda_t \ge h,\\
0, & \text{else}.
\end{cases}
\]

Outputs: \(\mathrm{score}_t = \Lambda_t\), \(\mathrm{residual}_t = \rho_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class KalmanInnovationResidualFOCuS` (`KalmanInnovationZScore innov_` + `FOCuS focus_`).
Result type is `InnovationChangepointResult`.

## Reference

- [Romano et al., FOCuS (arXiv:2110.08205)](https://arxiv.org/abs/2110.08205)
- [Welch & Bishop, “An Introduction to the Kalman Filter”](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
