# HilbertDominantCyclePeriod

## Summary

`HilbertDominantCyclePeriod` is RTTA's streaming estimate of the dominant market
cycle period using John Ehlers' Hilbert-transform adaptive method as ported from
TA-Lib `HT_DCPERIOD`. The output is a smoothed period (in bars), typically
clamped to the range \([6, 50]\).

## Update API

```python
result = rtta.HilbertDominantCyclePeriod(fillna=True).update(value)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fillna`  | `True`  | If `False`, NaN until TA-Lib lookback of 32 samples is exceeded |

`update(value)` consumes one scalar price and returns the current smoothed
dominant-cycle period. There are no free period parameters; the engine is fully
adaptive.

## Theory Of Operation

All Hilbert family indicators share `HilbertCycleEngine` in
`src/rtta/indicator.cpp`. Each bar:

1. **4-bar WMA smooth** of price:
   \((4x_t + 3x_{t-1} + 2x_{t-2} + x_{t-3})/10\).
2. **Odd/even Hilbert detrender** on the smooth using fixed FIR-style Hilbert
   taps \(a=0.0962\), \(b=0.5769\), scaled by the adaptive factor
   \(0.075\cdot\text{period} + 0.54\).
3. **In-phase / quadrature** components are advanced on alternating parity bars;
   further Hilbert stages produce \(I_2,Q_2\).
4. **Complex autocorrelation** updates real/imaginary parts \(Re,Im\); the
   instantaneous period is \(360^\circ / \mathrm{atan2}\)-style angle from
   \(Im/Re\), then rate-limited, hard-clamped to \([6,50]\), and EMA-smoothed.

`HilbertDominantCyclePeriod` exposes only the final **smoothed period**
`smooth_period_`.

## Recurrence

### Price smooth

\[
S_t = \frac{4 x_t + 3 x_{t-1} + 2 x_{t-2} + x_{t-3}}{10}
\]

### Adaptive Hilbert scale

\[
\lambda_t = 0.075\, P_{t-1} + 0.54
\]

Detrender / quadrature stages apply the TA-Lib odd-even Hilbert operator
\(H_a(\cdot)\) (coefficients \(a=0.0962\), \(b=0.5769\), 3-slot circular
buffers) and multiply by \(\lambda_t\).

### Homodyne discriminator

With smoothed complex components \(I2_t,Q2_t\):

\[
Re_t = 0.8\, Re_{t-1} + 0.2\,(I2_t I2_{t-1} + Q2_t Q2_{t-1})
\]

\[
Im_t = 0.8\, Im_{t-1} + 0.2\,(I2_t Q2_{t-1} - Q2_t I2_{t-1})
\]

\[
\tilde{P}_t =
\begin{cases}
\dfrac{360}{\operatorname{atan}(Im_t/Re_t)\cdot\frac{180}{\pi}} & Re_t,Im_t \ne 0 \\
P_{t-1} & \text{otherwise}
\end{cases}
\]

(Implementation uses `atan(im/re) * rad2deg` with `rad2deg = 45/atan(1)`.)

### Clamp and smooth

\[
\tilde{P}_t \leftarrow \operatorname{clip}\!\big(\tilde{P}_t,\; 0.67 P_{t-1},\; 1.5 P_{t-1}\big)
\]

\[
\tilde{P}_t \leftarrow \operatorname{clip}(\tilde{P}_t,\; 6,\; 50)
\]

\[
P_t = 0.2\,\tilde{P}_t + 0.8\, P_{t-1}
\]

\[
\overline{P}_t = 0.33\, P_t + 0.67\, \overline{P}_{t-1}
\]

The returned value is \(\overline{P}_t\) (`smooth_period_`).

With `fillna=False`, output is NaN until more than 32 updates have completed
(TA-Lib `HT_DCPERIOD` lookback).

## Implementation Notes

- Thin wrapper around `HilbertCycleEngine::period()` in
  `src/rtta/indicator.cpp` (`class HilbertDominantCyclePeriod`).
- Shared state also drives phase, phasor, sine wave, trendline, and trend mode.
- Lookback constant: `lookback_period_ = 32`.

## Reference

- [TA-Lib HT_DCPERIOD](https://ta-lib.org/functions/ht_dcperiod)
- [Ehlers — Optimal Adaptive Averages (MESA)](https://www.mesasoftware.com/papers/OptimalAdaptiveAverage.pdf)
