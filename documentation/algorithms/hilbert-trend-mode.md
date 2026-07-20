# HilbertTrendMode

## Summary

`HilbertTrendMode` is RTTA's streaming trend-versus-cycle regime flag from the
Hilbert engine (TA-Lib `HT_TRENDMODE`). It returns `1` in trend mode and `0` in
cycle mode.

## Update API

```python
result = rtta.HilbertTrendMode(fillna=True).update(value)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fillna`  | `True`  | If `False`, NaN until lookback of 63 samples is exceeded |

`update(value)` returns `1.0` (trend) or `0.0` (cycle), or NaN during warm-up
when `fillna=False`.

## Theory Of Operation

Ehlers / TA-Lib classify each bar as **cycle** or **trend** using signals
already computed by the Hilbert engine:

1. **Sine / lead-sine cross** → force cycle mode and reset trend age.
2. **Short trend age** — if bars since last cycle event are fewer than half the
   smoothed period, remain in cycle mode.
3. **Orderly phase advance** — if phase increased by roughly one period step
   (\(0.67\) to \(1.5\) times \(360/\overline{P}\)), remain in cycle mode.
4. **Price far from trendline** — if the current WMA-smoothed price differs from
   the Hilbert trendline by at least \(1.5\%\) relative, force trend mode.

The default before overrides is trend (`1`). The result is a binary regime
label for strategy filters, not a continuous strength measure.

## Recurrence

Let \(sine_t\), \(lead_t\) be the Hilbert sine pair, \(\phi_t\) the DC phase,
\(\overline{P}_t\) the smoothed period, \(S_t\) the 4-bar WMA price, and
\(TL_t\) the Hilbert trendline. Maintain integer `days_in_trend`.

Initialize each bar with \(trend \leftarrow 1\).

**Cross → cycle, reset age:**

\[
\begin{aligned}
&\text{if } (sine_t > lead_t \land sine_{t-1} \le lead_{t-1}) \\
&\quad\text{or } (sine_t < lead_t \land sine_{t-1} \ge lead_{t-1}): \\
&\qquad days \leftarrow 0,\; trend \leftarrow 0
\end{aligned}
\]

Then \(days \leftarrow days + 1\).

**Young trend age → cycle:**

\[
\text{if } days < 0.5\,\overline{P}_t:\quad trend \leftarrow 0
\]

**Phase step in band → cycle:**

\[
\Delta\phi = \phi_t - \phi_{t-1}
\]

\[
\text{if } \overline{P}_t \ne 0 \text{ and }
0.67\cdot\frac{360}{\overline{P}_t} < \Delta\phi < 1.5\cdot\frac{360}{\overline{P}_t}:
\quad trend \leftarrow 0
\]

**Distance from trendline → trend:**

\[
\text{if } TL_t \ne 0 \text{ and }
\left|\frac{S_t - TL_t}{TL_t}\right| \ge 0.015:
\quad trend \leftarrow 1
\]

Returned value: \(trendmode_t = trend\) as a double (`0.0` or `1.0`).

With `fillna=False`, NaN until more than 63 updates have completed.

## Implementation Notes

- Thin wrapper around `HilbertCycleEngine::trendmode()` (`class HilbertTrendMode`).
- Depends on sine, phase, smooth price, and trendline paths inside the same
  `update` of the engine.
- Lookback: `lookback_phase_ = 63`.

## Reference

- [TA-Lib HT_TRENDMODE](https://ta-lib.org/functions/ht_trendmode)
- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
