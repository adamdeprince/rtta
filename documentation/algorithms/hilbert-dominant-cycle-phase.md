# HilbertDominantCyclePhase

## Summary

`HilbertDominantCyclePhase` is RTTA's streaming dominant-cycle phase in degrees,
porting TA-Lib `HT_DCPHASE`. Phase is recovered from a short DFT-style sum of
Hilbert-smoothed prices over the current dominant period, with TA-Lib lag and
quadrant corrections.

## Update API

```python
result = rtta.HilbertDominantCyclePhase(fillna=True).update(value)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fillna`  | `True`  | If `False`, NaN until lookback of 63 samples is exceeded |

`update(value)` returns the current DC phase in degrees.

## Theory Of Operation

After `HilbertCycleEngine` estimates the smoothed dominant period
\(\overline{P}_t\), it treats the last \(\lfloor\overline{P}_t+0.5\rfloor\)
(clamped to 1…50) smoothed prices as one cycle of a sine/cosine basis. The
argument of that complex sum is the cycle phase. TA-Lib then applies fixed
offsets (\(+90^\circ\), period lag \(360/\overline{P}\), half-plane flip when the
cosine sum is negative, wrap when phase \(>315^\circ\)) so the phase aligns with
their sine-wave and trend-mode logic.

Phase advances through a cycle as the market completes one dominant oscillation;
jumps or stalls indicate regime changes or unreliable period estimates.

## Recurrence

Let \(\overline{P}_t\) be the smoothed period from
[`HilbertDominantCyclePeriod`](hilbert-dominant-cycle-period.md), and

\[
N = \operatorname{clip}\!\big(\lfloor \overline{P}_t + 0.5 \rfloor,\; 1,\; 50\big).
\]

With ring buffer of smoothed prices \(S^{(0)},\ldots,S^{(49)}\) (index walking
backward from the current smooth index):

\[
Real = \sum_{i=0}^{N-1} \sin\!\Big(\frac{2\pi i}{N}\Big)\, S^{(-i)},\qquad
Imag = \sum_{i=0}^{N-1} \cos\!\Big(\frac{2\pi i}{N}\Big)\, S^{(-i)}
\]

Base angle (degrees, matching C++ `atan * rad2deg`):

\[
\phi \leftarrow
\begin{cases}
\operatorname{atan}(Real/Imag)\cdot\frac{180}{\pi} & |Imag| > 0 \\
\phi - 90 & |Imag|\le 0.01 \land Real < 0 \\
\phi + 90 & |Imag|\le 0.01 \land Real > 0
\end{cases}
\]

TA-Lib adjustments:

\[
\phi \leftarrow \phi + 90
\]

\[
\phi \leftarrow \phi + \frac{360}{\overline{P}_t}\quad (\overline{P}_t \ne 0)
\]

\[
\phi \leftarrow \phi + 180 \quad \text{if } Imag < 0
\]

\[
\phi \leftarrow \phi - 360 \quad \text{if } \phi > 315
\]

Returned value: \(\phi_t = \phi\).

With `fillna=False`, output is NaN until more than 63 updates have completed.

## Implementation Notes

- Thin wrapper around `HilbertCycleEngine::phase()` (`class HilbertDominantCyclePhase`).
- Lookback constant: `lookback_phase_ = 63`.
- Same engine instance state as the other Hilbert indicators; each public class
  owns its own engine.

## Reference

- [TA-Lib HT_DCPHASE](https://ta-lib.org/functions/ht_dcphase)
- [MESA Software — Ehlers Hilbert papers](https://www.mesasoftware.com/)
