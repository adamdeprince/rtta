# RainbowMovingAverage

## Summary

`RainbowMovingAverage` is RTTA's streaming implementation of Mel Widner's rainbow:
a stack of recursive simple moving averages. Each layer is an SMA of the previous
layer. The indicator returns the deepest layer, the envelope of all layers, and
the rainbow width.

## Update API

```python
result = rtta.RainbowMovingAverage(period=2, layers=10, fillna=True).update(price)
# result.outer, result.highest, result.lowest, result.mid, result.width
```

The `update(...)` call consumes one price observation. `advance(...)` uses the
same input when the caller wants to update state without materializing a Python
return value. Array `batch(...)` / `replay_update_outputs(...)` follow the same
multi-output path as other RTTA result structs.

## Theory Of Operation

Widner's rainbow builds a family of nested smoothers rather than a single MA.
Layer \(1\) is an SMA of price; layer \(k\) is an SMA of layer \(k-1\). Deeper
layers lag more, so the set of layers fans out when price trends and collapses
when price consolidates. RTTA keeps one causal `SMA` state per layer and, each
tick, feeds the previous layer's output into the next.

## Recurrence

Let \(x_t\) be the input price, \(p\) be `period`, and \(L\) be `layers`.

\[
S^{(1)}_t = \operatorname{SMA}_p(x_t), \qquad
S^{(k)}_t = \operatorname{SMA}_p\!\bigl(S^{(k-1)}_t\bigr)
\quad\text{for }k=2,\ldots,L
\]

\[
H_t = \max_{1\le k\le L} S^{(k)}_t, \qquad
L_t = \min_{1\le k\le L} S^{(k)}_t
\]

\[
\begin{aligned}
\operatorname{outer}_t &= S^{(L)}_t \\
\operatorname{mid}_t &= \tfrac12(H_t + L_t) \\
\operatorname{width}_t &= H_t - L_t
\end{aligned}
\]

With `fillna=False`, outputs are `NaN` until roughly \(p\cdot L\) samples have
been seen (each successive SMA adds lag). With `fillna=True`, partial SMA means
are emitted during warmup.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class RainbowMovingAverage`. Each layer is a nested `SMA` member; the result
struct fields are `outer`, `highest`, `lowest`, `mid`, and `width`.

## Reference

- [Mel Widner, "Rainbow Charts," *Technical Analysis of Stocks & Commodities*,
  July 1997 (PDF mirror)](https://c.mql5.com/forextsd/forum/64/rainbow_oscillator_-_original_article_-_mel_widner.pdf)
- [TradingPedia: Rainbow Oscillator](https://www.tradingpedia.com/forex-trading-indicators/rainbow-oscillator/)
