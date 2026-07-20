# HilbertPhasor

## Summary

`HilbertPhasor` is RTTA's streaming Hilbert in-phase and quadrature components,
porting TA-Lib `HT_PHASOR`. Together they form a complex analytic signal of the
detrended, smoothed price used inside the dominant-cycle estimator.

## Update API

```python
result = rtta.HilbertPhasor(fillna=True).update(value)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fillna`  | `True`  | If `False`, NaN until lookback of 32 samples is exceeded |

`update(...)` returns a result with:

- `inphase` — in-phase component \(I\)
- `quadrature` — quadrature component \(Q\)

`advance(value)` updates state; `last()` returns the cached result.

## Theory Of Operation

The Hilbert transform produces a \(90^\circ\) phase-shifted companion of a real
signal. Ehlers / TA-Lib apply a short FIR-style Hilbert operator to a 4-bar WMA
of price (the detrender), then take:

- **In-phase** as the detrender delayed by three adaptive Hilbert stages
  (`i1_for_even_prev3_` / `i1_for_odd_prev3_` depending on bar parity).
- **Quadrature** as the Hilbert transform of the detrender (\(Q1\)).

The complex pair \((I,Q)\) feeds the homodyne discriminator that estimates
dominant period. Plotting \(I\) vs \(Q\) traces a phasor diagram of the local
cycle.

## Recurrence

Shared front end with other Hilbert indicators (see
[`HilbertDominantCyclePeriod`](hilbert-dominant-cycle-period.md)):

\[
S_t = \frac{4 x_t + 3 x_{t-1} + 2 x_{t-2} + x_{t-3}}{10}
\]

\[
\lambda_t = 0.075\, P_{t-1} + 0.54
\]

Let \(D_t\) be the adaptive Hilbert detrender of \(S_t\), and \(Q1_t\) the
adaptive Hilbert transform of \(D_t\). With bar parity alternating odd/even
filter banks:

\[
I_t = D_{t-3}^{\text{(parity path)}},\qquad
Q_t = Q1_t
\]

In code after each update:

\[
\texttt{inphase} = I_t,\qquad
\texttt{quadrature} = Q_t
\]

With `fillna=False`, both fields are NaN until more than 32 samples have been
processed.

## Implementation Notes

- Implemented as `class HilbertPhasor` wrapping `HilbertCycleEngine::inphase()`
  and `::quadrature()`.
- Result type: `HilbertPhasorResult` (`inphase`, `quadrature`).
- Lookback matches `HT_DCPERIOD` / `HT_PHASOR` (32).
- Batch helper: `batch_hilbert_phasor`.

## Reference

- [TA-Lib HT_PHASOR](https://ta-lib.org/functions/ht_phasor)
- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
