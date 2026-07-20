# HilbertSineWave

## Summary

`HilbertSineWave` is RTTA's streaming Hilbert sine and lead-sine pair from the
dominant-cycle phase (TA-Lib `HT_SINE`). The two waves form a cycle oscillator
suitable for identifying cyclic turns when they cross.

## Update API

```python
result = rtta.HilbertSineWave(fillna=True).update(value)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fillna`  | `True`  | If `False`, NaN until lookback of 63 samples is exceeded |

`update(...)` returns:

- `sine` — \(\sin(\phi_t)\)
- `lead_sine` — \(\sin(\phi_t + 45^\circ)\)

where \(\phi_t\) is the dominant-cycle phase in degrees.
`advance(value)` updates state; `last()` returns the cached result.

## Theory Of Operation

Once [`HilbertDominantCyclePhase`](hilbert-dominant-cycle-phase.md) produces
phase \(\phi_t\), mapping through sine yields a normalized cycle waveform on
\([-1,1]\). The lead sine is advanced by \(45^\circ\) so that crosses of sine and
lead sine occur ahead of pure sine peaks and troughs—TA-Lib / Ehlers use these
crosses as cycle-mode timing events (also consumed by
[`HilbertTrendMode`](hilbert-trend-mode.md)).

Because phase is derived from the adaptive period, the sine wave stretches and
compresses as the estimated cycle length changes.

## Recurrence

Let \(\phi_t\) be the DC phase from the shared engine (degrees). Convert with
`deg2rad = 1/rad2deg`, `rad2deg = 45/\operatorname{atan}(1)`:

\[
sine_t = \sin(\phi_t \cdot \texttt{deg2rad})
\]

\[
lead\_sine_t = \sin\big((\phi_t + 45)\cdot \texttt{deg2rad}\big)
\]

With `fillna=False`, both fields are NaN until more than 63 samples have been
processed.

## Implementation Notes

- Implemented as `class HilbertSineWave` wrapping `HilbertCycleEngine::sine()`
  and `::lead_sine()`.
- Result type: `HilbertSineWaveResult` (`sine`, `lead_sine`).
- Lookback: `lookback_phase_ = 63` (same as phase / trendline / trend mode).
- Batch helper: `batch_hilbert_sine_wave`.

## Reference

- [TA-Lib HT_SINE](https://ta-lib.org/functions/ht_sine)
- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
