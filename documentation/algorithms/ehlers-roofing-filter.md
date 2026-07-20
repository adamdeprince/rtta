# EhlersRoofingFilter

## Summary

`EhlersRoofingFilter` is RTTA's streaming roofing filter: a two-pole high-pass
stage removes long-term drift, then an Ehlers Super Smoother low-pass retains
the mid-band cycle content. The result is a band-limited series useful as a
noise-reduced oscillator input.

## Update API

```python
result = rtta.EhlersRoofingFilter(hp_period=48, lp_period=10, fillna=True).update(price)
```

| Parameter   | Default | Meaning |
|-------------|---------|---------|
| `hp_period` | `48`    | High-pass critical period (minimum 2) |
| `lp_period` | `10`    | Super Smoother low-pass period (minimum 2) |
| `fillna`    | `True`  | If `False`, NaN until `hp_period` samples |

`update(...)` returns a result with:

- `roof` — band-passed (roofing) output
- `highpass` — intermediate high-pass series

`advance(price)` updates state without returning a new Python object; `last()`
returns the most recent result.

## Theory Of Operation

Price series mix a slow trend (very low frequency), intermediate cycles, and
high-frequency noise. A roofing filter:

1. **High-pass** with period near `hp_period` removes the trend so the residual
   oscillates around zero.
2. **Super Smoother low-pass** with period `lp_period` removes residual noise
   above the cycle band of interest.

The high-pass is Ehlers' two-pole form with coefficient \(\alpha\) derived from
angle \(0.707 \cdot 2\pi / P_{hp}\) (the \(0.707 \approx 1/\sqrt{2}\) factor
sets the two-pole high-pass response). The low-pass reuses the same Super
Smoother coefficient construction as [`EhlersSuperSmoother`](ehlers-super-smoother.md),
applied to the high-pass stream rather than raw price.

## Recurrence

### High-pass coefficients

Let \(P_h = \max(\texttt{hp\_period}, 2)\):

\[
\theta_h = \frac{0.707 \cdot 2\pi}{P_h},\qquad
\alpha = \frac{\cos\theta_h + \sin\theta_h - 1}{\cos\theta_h}
\]

### Super Smoother coefficients (low-pass)

Let \(P_\ell = \max(\texttt{lp\_period}, 2)\):

\[
\theta_\ell = \frac{\sqrt{2}\,\pi}{P_\ell},\qquad
a_1 = e^{-\theta_\ell},\qquad
b_1 = 2 a_1 \cos(\theta_\ell)
\]

\[
c_2 = b_1,\qquad
c_3 = -a_1^{2},\qquad
c_1 = 1 - c_2 - c_3
\]

### High-pass stage

For the first two samples, \(HP_t = 0\). Thereafter, with
\(k = 1 - \alpha/2\):

\[
\begin{aligned}
HP_t &= k^{2}\,(x_t - 2 x_{t-1} + x_{t-2}) \\
&\quad + 2(1-\alpha)\, HP_{t-1} \\
&\quad - (1-\alpha)^{2}\, HP_{t-2}
\end{aligned}
\]

### Roofing (Super Smoother of high-pass)

For the first two samples, \(R_t = HP_t\). Thereafter:

\[
R_t = c_1 \cdot \frac{HP_t + HP_{t-1}}{2} + c_2\, R_{t-1} + c_3\, R_{t-2}
\]

Result fields: `roof` \(= R_t\), `highpass` \(= HP_t\).

When `fillna=False` and fewer than \(P_h\) samples have been seen, both fields
are NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class EhlersRoofingFilter`.
- Result type: `EhlersRoofingFilterResult` (`roof`, `highpass`).
- High-pass and roof each keep two lags of state (`hp1_`/`hp2_`,
  `roof1_`/`roof2_`) and two lags of price for the second difference.
- Batch helper: `batch_ehlers_roofing`.

## Reference

- [MESA Software — John Ehlers papers](https://www.mesasoftware.com/)
- [Ehlers roofing / band-pass filter discussion](https://www.mesasoftware.com/papers/GaussianFilters.pdf)
