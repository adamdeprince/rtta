# EhlersCyberCycle

## Summary

`EhlersCyberCycle` is RTTA's streaming implementation of John Ehlers' cyber-cycle
oscillator. Price is lightly smoothed, then a two-pole high-pass extracts the
dominant cycle component. A one-bar lag of the cycle is returned as a trigger
line for cross-based timing.

## Update API

```python
result = rtta.EhlersCyberCycle(period=20, fillna=True).update(price)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `period`  | `20`    | High-pass critical period (minimum 2) |
| `fillna`  | `True`  | If `False`, NaN until `period` samples |

`update(...)` returns:

- `cycle` — current cyber-cycle value
- `trigger` — previous bar's cycle (one-bar lag used as trigger)

`advance(price)` updates state; `last()` returns the cached result.

## Theory Of Operation

The cyber cycle is designed to isolate cyclic content while cancelling trend.
RTTA follows Ehlers' practical streaming form:

1. **4-bar weighted smooth** of price (once enough history exists):
   \((x_t + 2x_{t-1} + 2x_{t-2} + x_{t-3})/6\).
2. **Two-pole high-pass** of that smooth with coefficient \(\alpha\) from the
   critical period (same angle construction as the roofing high-pass,
   \(0.707\cdot 2\pi/P\)).
3. During a short bootstrap (first 7 samples), a simple second-difference
   proxy is used so the filter is defined before full recursive state exists.
4. The **trigger** is the previous cycle value, so crossings of cycle vs trigger
   mark short-term cycle turns.

Positive cycle values indicate the cyclic component is above its local zero;
negative values indicate the opposite.

## Recurrence

### Alpha

Let \(P = \max(\texttt{period}, 2)\):

\[
\theta = \frac{0.707 \cdot 2\pi}{P},\qquad
\alpha = \frac{\cos\theta + \sin\theta - 1}{\cos\theta}
\]

### Price smooth

For sample index \(t \ge 3\) (zero-based count \(\ge 3\)):

\[
s_t = \frac{x_t + 2 x_{t-1} + 2 x_{t-2} + x_{t-3}}{6}
\]

Otherwise \(s_t = x_t\).

### Cycle bootstrap (\(t < 7\))

\[
C_t = \frac{x_t - 2 x_{t-1} + x_{t-2}}{4}
\]

### Cycle recursion (\(t \ge 7\))

Let \(a = 1 - \alpha/2\):

\[
\begin{aligned}
C_t &= a^{2}\,(s_t - 2 s_{t-1} + s_{t-2}) \\
&\quad + 2(1-\alpha)\, C_{t-1} \\
&\quad - (1-\alpha)^{2}\, C_{t-2}
\end{aligned}
\]

### Trigger

\[
T_t = C_{t-1}
\]

(in code, `trigger = c1_` is the cycle value **before** writing the new cycle
into state).

Result: `cycle` \(= C_t\), `trigger` \(= T_t\).

When `fillna=False` and fewer than \(P\) samples have been processed, both
fields are NaN.

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class EhlersCyberCycle`.
- Result type: `EhlersCyberCycleResult` (`cycle`, `trigger`).
- Price lags `p1_`,`p2_`,`p3_`; smooth lags `s1_`,`s2_`; cycle lags `c1_`,`c2_`.
- Batch helper: `batch_ehlers_cyber_cycle`.

## Reference

- [MESA Software — Ehlers papers](https://www.mesasoftware.com/)
- [Cyber Cycle background](https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf)
