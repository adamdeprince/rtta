# EhlersCenterOfGravity

## Summary

`EhlersCenterOfGravity` is RTTA's streaming center-of-gravity (CG) oscillator
from John Ehlers. Over a rolling price window it measures the "balance point"
of the price path and recenters it so the oscillator fluctuates about zero.
The prior CG is returned as `lag` for trigger-style crosses.

## Update API

```python
result = rtta.EhlersCenterOfGravity(window=10, fillna=True).update(price)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window`  | `10`    | Rolling lookback (minimum 2) |
| `fillna`  | `True`  | If `False`, NaN until the window is full |

`update(...)` returns:

- `cg` — current center-of-gravity oscillator
- `lag` — previous bar's `cg` (for cross signals)

`advance(price)` updates state; `last()` returns the cached result.

## Theory Of Operation

Treat the last \(n\) prices as point masses of equal mass along a bar axis.
Ehlers assigns weight \(1\) to the **newest** bar and weight \(n\) to the
**oldest**, so the first moment is biased toward older prices when the market
has been falling into the window and toward newer prices when it has been
rising. Dividing by the sum of prices (not sum of weights) yields a raw CG
coordinate; subtracting \((n+1)/2\) centers the null at the geometric mid-window
so the oscillator is roughly zero-mean.

Because CG leads ordinary smoothed oscillators at turning points, Ehlers often
plots CG against its one-bar lag; RTTA exposes that lag as the `lag` field.

## Recurrence

Let \(n\) be the current number of buffered prices (\(n \le W\), \(W\) the
constructor window). Index \(i=0\) as the **newest** sample and \(i=n-1\) as
the oldest. Weights \(w_i = i+1\) (newest weight 1, oldest weight \(n\)):

\[
N_t = \sum_{i=0}^{n-1} w_i\, x_{t-i},\qquad
D_t = \sum_{i=0}^{n-1} x_{t-i}
\]

\[
CG_t =
\begin{cases}
0 & \text{if } D_t = 0 \\
-\dfrac{N_t}{D_t} + \dfrac{n+1}{2} & \text{otherwise}
\end{cases}
\]

\[
lag_t = CG_{t-1}
\]

(with \(lag\) for the first bar equal to the constructor seed `0.0` before any
update).

When `fillna=False` and the rolling buffer is not yet full, both fields are
NaN. When `fillna=True`, partial windows use the available \(n\) samples with
the same weight convention.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class EhlersCenterOfGravity`.
- Result type: `EhlersCenterOfGravityResult` (`cg`, `lag`).
- Uses `RollingBuffer prices_`; newest is `prices_.at(n-1)`, matching the
  loop `prices_.at(n - 1 - i)` for weight index \(i\).
- Batch helper: `batch_ehlers_cg`.

## Reference

- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
- [Ehlers CG / stochastic CG discussion](https://www.mesasoftware.com/papers/TheCGOscillator.pdf)
