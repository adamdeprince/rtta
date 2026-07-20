# Alligator

## Summary

`Alligator` is RTTA's streaming Bill Williams Alligator: three shifted smoothed
moving averages of median price—**jaw**, **teeth**, and **lips**—that open and
close like an alligator's mouth as trend and consolidation alternate.

## Update API

```python
result = rtta.Alligator(
    jaw_window=13, teeth_window=8, lips_window=5,
    jaw_shift=8, teeth_shift=5, lips_shift=3,
    fillna=True,
).update(high, low)
```

| Parameter      | Default | Meaning |
|----------------|---------|---------|
| `jaw_window`   | `13`    | SMMA length for jaw (blue) |
| `teeth_window` | `8`     | SMMA length for teeth (red) |
| `lips_window`  | `5`     | SMMA length for lips (green) |
| `jaw_shift`    | `8`     | Forward shift (bars of delay) for jaw |
| `teeth_shift`  | `5`     | Delay for teeth |
| `lips_shift`   | `3`     | Delay for lips |
| `fillna`       | `True`  | If `False`, NaN while any line is still NaN from shift warm-up |

`update(high, low)` returns `jaw`, `teeth`, `lips`.
`advance(...)` updates state; `last()` returns the cached result.

## Theory Of Operation

Median price \(m_t=(h_t+l_t)/2\) is smoothed with three independent Wilder SMMA
(RMA) windows. On classical charts the lines are *drawn forward* (future
shifted). In a causal streaming API that future shift is implemented as a
**delay buffer** of the same length: the value reported today is the SMMA from
`shift` bars ago, matching how a completed chart would have shown that SMMA
aligned with the current bar after the forward offset.

- **Lips** (fastest) react first when a trend starts.
- **Teeth** form the middle balance line.
- **Jaw** (slowest) is the sleeping / deep trend line.

When the lines are intertwined (mouth closed), the "alligator sleeps"
(range/chop). When they fan out in order, the mouth is open and a trend is
active. [`GatorOscillator`](gator-oscillator.md) quantifies the gaps between
these lines.

## Recurrence

\[
m_t = \frac{h_t + l_t}{2}
\]

Let \(\operatorname{SMMA}_n\) be [`SmoothedMovingAverage`](smoothed-moving-average.md)
(SMA seed, then Wilder recursion with \(\alpha=1/n\)). Let \(\Delta_k\) be a
causal delay of \(k\) bars (`ShiftBuffer`: returns NaN until \(k\) samples have
entered, then the value from \(k\) bars ago).

\[
jaw_t = \Delta_{\texttt{jaw\_shift}}\big(\operatorname{SMMA}_{\texttt{jaw\_window}}(m)\big)_t
\]

\[
teeth_t = \Delta_{\texttt{teeth\_shift}}\big(\operatorname{SMMA}_{\texttt{teeth\_window}}(m)\big)_t
\]

\[
lips_t = \Delta_{\texttt{lips\_shift}}\big(\operatorname{SMMA}_{\texttt{lips\_window}}(m)\big)_t
\]

Defaults: jaw \(13/8\), teeth \(8/5\), lips \(5/3\).

When `fillna=False` and any of the three delayed lines is still NaN, the result
struct is all-NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class Alligator`.
- SMMA instances always use `fillna=True` so partial SMAs exist; shifts produce
  NaN until full.
- Result type: `AlligatorResult` (`jaw`, `teeth`, `lips`).
- Batch helper: `batch_alligator`.

## Reference

- [Investopedia — Alligator Indicator](https://www.investopedia.com/articles/trading/06/alligator.asp)
- [Bill Williams Alligator overview](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/gator-oscillator)
