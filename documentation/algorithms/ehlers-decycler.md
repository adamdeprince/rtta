# EhlersDecycler

## Summary

`EhlersDecycler` is RTTA's streaming Ehlers decycler. It estimates a
low-frequency **decycle** component of price (trend-like smooth) and an
**oscillator** residual \(x_t - \text{decycle}_t\) that emphasizes what was
removed—the higher-frequency content.

## Update API

```python
result = rtta.EhlersDecycler(hp_period=60, fillna=True).update(price)
```

| Parameter   | Default | Meaning |
|-------------|---------|---------|
| `hp_period` | `60`    | Period controlling the high-pass / decycle cutoff (minimum 2) |
| `fillna`    | `True`  | If `False`, NaN until `hp_period` samples |

`update(...)` returns:

- `decycle` — low-frequency decycle estimate
- `oscillator` — \(price - decycle\)

`advance(price)` updates state; `last()` returns the cached result.

## Theory Of Operation

Ehlers' decycler is obtained by subtracting a one-pole high-pass from price
(equivalently, a complementary low-pass). With high-pass coefficient \(\alpha\)
derived from period \(P\), the low-frequency path is:

\[
D_t = \frac{\alpha}{2}(x_t + x_{t-1}) + (1-\alpha) D_{t-1}
\]

so \(D_t\) tracks slow structure while the residual \(x_t - D_t\) behaves like a
zero-mean oscillator of faster swings. Larger `hp_period` pushes more of the
spectrum into the decycle and leaves a quieter oscillator; smaller periods make
the decycle hug price more tightly.

This is simpler than the two-pole roofing filter: a single \(\alpha\) and one
lag of price/decycle state.

## Recurrence

Let \(P = \max(\texttt{hp\_period}, 2)\):

\[
\theta = \frac{2\pi}{P},\qquad
\alpha = \frac{\cos\theta + \sin\theta - 1}{\cos\theta}
\]

First sample (\(t = 0\)):

\[
D_0 = x_0
\]

Thereafter:

\[
D_t = \frac{\alpha}{2}\,(x_t + x_{t-1}) + (1-\alpha)\, D_{t-1}
\]

\[
O_t = x_t - D_t
\]

Result: `decycle` \(= D_t\), `oscillator` \(= O_t\).

When `fillna=False` and fewer than \(P\) samples have been processed, both
fields are NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class EhlersDecycler`.
- Result type: `EhlersDecyclerResult` (`decycle`, `oscillator`).
- State: `price1_`, `decycle1_`, sample `count_`.
- Note: \(\theta = 2\pi/P\) (no \(0.707\) factor), unlike
  [`EhlersRoofingFilter`](ehlers-roofing-filter.md) / cyber-cycle high-pass.
- Batch helper: `batch_ehlers_decycler`.

## Reference

- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
- [Decycler / high-pass complementary filters](https://www.mesasoftware.com/papers/GaussianFilters.pdf)
