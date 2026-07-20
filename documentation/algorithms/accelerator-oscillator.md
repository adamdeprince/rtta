# AcceleratorOscillator

## Summary

`AcceleratorOscillator` is RTTA's streaming Bill Williams Accelerator Oscillator
(AC): the Awesome Oscillator minus a short SMA of itself. It measures the
*change* in driving force of the market (acceleration), not just the force.

## Update API

```python
result = rtta.AcceleratorOscillator(ao_slow=34, ao_fast=5, smooth=5, fillna=True).update(high, low)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ao_slow` | `34`    | Slow SMA length inside Awesome Oscillator |
| `ao_fast` | `5`     | Fast SMA length inside Awesome Oscillator |
| `smooth`  | `5`     | SMA length applied to AO |
| `fillna`  | `True`  | If `False`, NaN until warm-up length |

`update(high, low)` returns a scalar AC value.

## Theory Of Operation

Bill Williams' Awesome Oscillator (AO) is the difference between a fast and slow
SMA of median price \((h+l)/2\). The Accelerator Oscillator is AO minus a
further SMA of AO:

\[
AC_t = AO_t - \operatorname{SMA}(AO)_t.
\]

When AC is above zero and rising, upside force is accelerating; when below zero
and falling, downside force is accelerating. Zero-line and color-change rules
from Williams' framework treat AC as a confirmation filter for AO and Alligator
setups.

RTTA composes the existing [`AwesomeOscillator`](awesome-oscillator.md) with an
SMA of length `smooth`.

## Recurrence

Median price and AO (matching `AwesomeOscillator`; high/low are ordered so
\(h\ge l\)):

\[
m_t = \frac{h_t + l_t}{2}
\]

\[
AO_t = \operatorname{SMA}_{n_f}(m_t) - \operatorname{SMA}_{n_s}(m_t)
\]

with defaults \(n_f=5\), \(n_s=34\).

Let \(k=\texttt{smooth}\) (default 5). Feed AO into an SMA (NaN AO treated as
`0.0` when updating the SMA so state advances):

\[
\overline{AO}_t = \operatorname{SMA}_k(AO_t)
\]

\[
AC_t = AO_t - \overline{AO}_t
\]

Warm-up when `fillna=False`: require
\(\max(n_s,n_f) + k\) samples before a non-NaN return. If AO itself is still
NaN, return `0.0` when `fillna=True`, else NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class AcceleratorOscillator`.
- Internals: `AwesomeOscillator ao_` with `fillna=True`, plus `SMA ao_sma_`.
- Constructor order is `(ao_slow, ao_fast, smooth)` matching AO's
  `(window_1=slow, window_2=fast)`.

## Reference

- [Investopedia — Bill Williams Alligator / AC context](https://www.investopedia.com/articles/trading/06/alligator.asp)
- [Awesome Oscillator background](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/awesome-oscillator-ao)
