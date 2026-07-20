# EhlersSuperSmoother

## Summary

`EhlersSuperSmoother` is RTTA's streaming two-pole Super Smoother low-pass filter
by John F. Ehlers. It attenuates high-frequency market noise with substantially
less lag than a comparable exponential moving average while remaining causal and
suitable for bar-by-bar streaming use.

## Update API

```python
result = rtta.EhlersSuperSmoother(period=10, fillna=True).update(price)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `period`  | `10`    | Critical period \(P\) of the low-pass (minimum 2) |
| `fillna`  | `True`  | If `False`, return NaN until `period` samples have been seen |

The `update(...)` call consumes one scalar price observation and returns the
current filtered value. `advance(...)` is not exposed on this class; use
`update` or `batch` / `batch_array` for bulk series.

## Theory Of Operation

Ehlers designs the Super Smoother as a two-pole recursive low-pass whose poles
lie on a circle set by the critical period \(P\). The continuous-time prototype
uses \(\sqrt{2}\,\pi / P\) as the angle of the complex conjugate pole pair; the
discrete filter coefficients \(c_1,c_2,c_3\) follow from the exponential and
cosine of that angle, with \(c_1\) chosen so DC gain is unity.

The input is a two-sample average \((x_t + x_{t-1})/2\) before the recursive
section, which reduces the one-bar stair-step of discrete price. The first two
samples seed the filter state to the raw price so the recurrence has finite
history without a long warm-up buffer.

Compared with an EMA of similar noise rejection, Super Smoother rolls off faster
past the critical period and produces a cleaner smooth for cycle indicators that
chain after it (for example the roofing filter low-pass stage).

## Recurrence

Let \(x_t\) be the input price and \(P=\max(\texttt{period},2)\) the critical
period. Precompute once in the constructor:

\[
\theta = \frac{\sqrt{2}\,\pi}{P},\qquad
a_1 = e^{-\theta},\qquad
b_1 = 2 a_1 \cos(\theta)
\]

\[
c_2 = b_1,\qquad
c_3 = -a_1^{2},\qquad
c_1 = 1 - c_2 - c_3
\]

State after each bar: previous price \(x_{t-1}\) and previous two filter outputs
\(y_{t-1}\), \(y_{t-2}\). For the first two samples (\(t < 2\)):

\[
y_t = x_t
\]

Thereafter:

\[
y_t = c_1 \cdot \frac{x_t + x_{t-1}}{2} + c_2\, y_{t-1} + c_3\, y_{t-2}
\]

When `fillna=False` and fewer than \(P\) samples have been processed, the return
value is NaN; otherwise \(y_t\) is returned.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class EhlersSuperSmoother`.
- Coefficients use the constant \(\sqrt{2}\approx 1.4142135623730951\) and
  \(\pi\) matching the C++ source.
- State variables: `price1_`, `filt1_`, `filt2_`, sample `count_`.
- Output is a scalar `double`, not a result struct.

## Reference

- [Ehlers — Super Smoother (MESA Software papers)](https://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf)
- [Ehlers filter overview (Rocket Science for Traders / cycle tools)](https://www.mesasoftware.com/)
