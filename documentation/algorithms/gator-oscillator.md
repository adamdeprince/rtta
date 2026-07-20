# GatorOscillator

## Summary

`GatorOscillator` is RTTA's streaming Bill Williams Gator Oscillator. It plots
the absolute gaps between Alligator jaw–teeth (upper histogram) and
teeth–lips (lower histogram, signed negative) as a visual of whether the
Alligator mouth is opening or closing.

## Update API

```python
result = rtta.GatorOscillator(
    jaw_window=13, teeth_window=8, lips_window=5,
    jaw_shift=8, teeth_shift=5, lips_shift=3,
    fillna=True,
).update(high, low)
```

Parameters match [`Alligator`](alligator.md) (same defaults for windows and
shifts).

`update(high, low)` returns:

- `upper` — \(|jaw - teeth|\)
- `lower` — \(-|teeth - lips|\)

`advance(...)` updates state; `last()` returns the cached result.

## Theory Of Operation

The Gator does not introduce new smoothers; it is a pure transform of Alligator
lines. Large absolute gaps mean the mouth is open (trend). Small gaps mean the
mouth is closed (sleep / consolidation). The lower series is drawn below zero by
convention so a single histogram pane shows both gaps.

Williams' color rules (expansion vs contraction of each gap bar-over-bar) can be
applied by the caller from successive `upper`/`lower` values; RTTA returns the
raw signed magnitudes only.

## Recurrence

Let \((J_t, T_t, L_t)\) be the Alligator jaw, teeth, and lips at time \(t\):

\[
upper_t = |J_t - T_t|
\]

\[
lower_t = -|T_t - L_t|
\]

If any Alligator component is NaN, both fields are NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class GatorOscillator`.
- Owns an internal `Alligator` with identical constructor parameters.
- Result type: `GatorOscillatorResult` (`upper`, `lower`).
- Batch helper: `batch_gator`.

## Reference

- [StockCharts — Gator Oscillator](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/gator-oscillator)
- [Investopedia — Alligator / Gator context](https://www.investopedia.com/articles/trading/06/alligator.asp)
