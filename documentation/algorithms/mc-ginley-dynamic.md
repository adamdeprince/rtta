# McGinleyDynamic

## Summary

`McGinleyDynamic` is RTTA's streaming McGinley Dynamic: an adaptive moving
average that automatically speeds up when price runs away from the line and
slows when price is nearby, reducing whipsaw relative to a fixed-speed EMA.

## Update API

```python
result = rtta.McGinleyDynamic(window=14, fillna=True).update(price)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window`  | `14`    | Speed constant \(N\) |
| `fillna`  | `True`  | If `False`, NaN until `window` samples |

`update(price)` returns the current McGinley Dynamic value as a scalar.

## Theory Of Operation

John R. McGinley designed the Dynamic to track price more closely than simple or
exponential averages without constant period retuning. The step toward price is
divided by \(N (price/MD)^{4}\):

- When price is far above MD, the ratio \(>1\) and the denominator grows, but
  the large numerator still pulls MD up; the fourth-power response is tuned so
  the line "catches up" smoothly rather than lagging like a slow SMA.
- When price oscillates near MD, effective smoothing is heavier, limiting noise.

The first observation seeds \(MD = price\). A zero MD is reseeded to price; a
non-finite denominator also reseeds.

## Recurrence

Let \(z_t\) be price and \(N=\max(\texttt{window},1)\). Seed:

\[
MD_1 = z_1
\]

For \(t > 1\), if \(MD_{t-1} = 0\) set \(MD_t = z_t\); else with
\(r_t = z_t / MD_{t-1}\) (safe-divide defaulting to 1):

\[
D_t = N \cdot r_t^{4}
\]

\[
MD_t =
\begin{cases}
MD_{t-1} + \dfrac{z_t - MD_{t-1}}{D_t} & D_t \ne 0 \land D_t \text{ finite} \\
z_t & \text{otherwise}
\end{cases}
\]

When `fillna=False` and fewer than \(N\) samples have been seen, return NaN;
otherwise return \(MD_t\).

## Implementation Notes

- Implemented in `src/rtta/indicator.cpp` in `class McGinleyDynamic`.
- Uses `safe_divide(price, value_, 1.0)` for the ratio.
- Output is a scalar `double`.

## Reference

- [Investopedia — McGinley Dynamic](https://www.investopedia.com/terms/m/mcginley-dynamic.asp)
- [McGinley Dynamic overview](https://www.tradingview.com/support/solutions/43000589132-mcginley-dynamic/)
