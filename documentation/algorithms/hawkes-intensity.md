# HawkesIntensity

## Summary

`HawkesIntensity` is RTTA's streaming exponential Hawkes process intensity for
event times: a constant baseline plus a self-exciting residual that decays
exponentially between events and jumps when new events arrive.

## Update API

```python
result = rtta.HawkesIntensity(
    mu=1.0, alpha=0.5, beta=1.0, fillna=True
).update(time, jump=1.0)
# result.intensity, result.excitation, result.baseline
```

`time` is a real-valued event clock (seconds, exchange timestamps, or any
monotonic scale consistent with `beta`). `jump` is the mark size of the event
(default `1.0`). `batch(time, jump)` or `batch(time)` (unit jumps) return
multi-output arrays.

## Theory Of Operation

A univariate exponential Hawkes process has conditional intensity

\[
\lambda(t) = \mu + \sum_{t_i < t} \alpha\, e^{-\beta(t-t_i)}
\]

with baseline \(\mu\), excitation amplitude \(\alpha\), and decay \(\beta\).
RTTA maintains the residual excitation \(A_t\) in a recursive form so each
update is \(O(1)\): decay the previous excitation by \(e^{-\beta\Delta t}\),
then add \(\alpha\cdot\operatorname{jump}\). Clustering of events raises
intensity above baseline; quiet periods pull it back toward \(\mu\).

## Recurrence

State is residual excitation \(A\) and last event time \(t_{\mathrm{last}}\).
On the first event at time \(t\) with jump \(j\):

\[
A \leftarrow \alpha j, \qquad
\lambda = \mu + A
\]

On a later event at time \(t\) with jump \(j\) and \(\Delta t = t - t_{\mathrm{last}}\):

\[
A \leftarrow A\, e^{-\beta\,\lvert\Delta t\rvert} + \alpha j
\]

\[
\lambda_t = \mu + A, \qquad
\operatorname{excitation}_t = A, \qquad
\operatorname{baseline}_t = \mu
\]

(If time goes backward, RTTA still decays with \(\lvert\Delta t\rvert\) and
applies the jump so the stream remains defined.)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class HawkesIntensity`. Parameters are floored so \(\mu,\alpha\ge 0\) and
\(\beta > 0\).

## Reference

- [Hawkes, "Spectra of some self-exciting and mutually exciting point
  processes," *Biometrika*, 1971](https://doi.org/10.1093/biomet/58.1.83)
- [Bacry, Mastromatteo, Muzy, "Hawkes processes in finance,"
  arXiv:1502.04592](https://arxiv.org/abs/1502.04592)
