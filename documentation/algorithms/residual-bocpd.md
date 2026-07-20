# ResidualBOCPD

## Summary

`ResidualBOCPD` runs **bounded-memory Bayesian online changepoint detection** (Adams &
MacKay style) on a residual/innovation series. It returns a binary `signal` when the
posterior mass on a new run length exceeds a threshold, plus that change probability.

## Update API

```python
import rtta

ind = rtta.ResidualBOCPD(
    max_run_length=128, hazard=0.01, threshold=0.5, min_variance=1e-6
)
result = ind.update(residual)
# result.signal ∈ {0, 1}, result.probability  # P(run length = 0) after update
```

`advance(...)` updates state without returning a result.
`last_probability()` exposes the same probability as `result.probability`.

## Theory Of Operation

BOCPD maintains a posterior over the time since the last changepoint (run length). Under
a constant hazard rate \(h\), run lengths grow with probability \(1-h\) or reset with
probability \(h\). Observation likelihoods are Gaussian with online mean/variance per
run-length hypothesis. Bounding the support to \(\{0,\ldots,R_{\max}\}\) keeps memory and
time \(O(R_{\max})\) per tick.

Applied to residuals, a high \(P(r_t=0)\) indicates the data prefer starting a new regime
relative to continuing previous residual statistics—i.e., a model-based changepoint.

## Recurrence

Let \(R_{\max}\) be `max_run_length`, hazard \(h\), threshold \(\tau\). Maintain for each
run length \(r \in \{0,\ldots,R_{\max}\}\) a probability \(\pi_t(r)\), mean \(\mu_t(r)\),
variance \(v_t(r)\), and count \(n_t(r)\).

First observation: \(\pi_0(0)=1\), \(\mu_0(0)=r_0\), \(v_0(0)=0\), \(n_0(0)=1\); signal 0.

Thereafter, for each active run \(r\) with \(\pi_{t-1}(r)>0\):

\[
L_r = \mathcal{N}\!\left(r_t;\, \mu_{t-1}(r),\, \max(v_{t-1}(r), v_{\min})\right),
\quad
w_r = \pi_{t-1}(r)\, L_r.
\]

Growth and change messages:

\[
\tilde{\pi}_t(\min(r+1, R_{\max})) \mathrel{+}= w_r (1-h),
\quad
\tilde{\pi}_t(0) \mathrel{+}= w_r\, h.
\]

Run statistics on growth use Welford-style updates with count
\(n' = \min(n_{t-1}(r)+1, R_{\max})\):

\[
\mu' = \mu + \frac{r_t - \mu}{n'},\qquad
v' = \bigl(1 - \tfrac{1}{n'}\bigr)\bigl(v + \tfrac{1}{n'}(r_t-\mu)^2\bigr).
\]

Changepoint mass at run 0 is initialized with \(\mu=r_t\), \(v=0\), \(n=1\). Normalize
\(\tilde{\pi}_t\) to \(\pi_t\). Outputs:

\[
\mathrm{probability}_t = \pi_t(0),\qquad
\mathrm{signal}_t = \mathbf{1}\{\pi_t(0) \ge \tau\}.
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ResidualBOCPD`, which
wraps `BoundedBOCPD` (`core_.update` / `last_probability`).

## Reference

- [Adams & MacKay, “Bayesian Online Changepoint Detection” (arXiv:0710.3742)](https://arxiv.org/abs/0710.3742)
