# RainbowOscillator

## Summary

`RainbowOscillator` is RTTA's streaming implementation of Mel Widner's rainbow
oscillator: the percent width of the recursive SMA rainbow relative to price,
plus the percent position of price inside that band.

## Update API

```python
result = rtta.RainbowOscillator(period=2, layers=10, fillna=True).update(price)
# result.value, result.position, result.width
```

The `update(...)` call consumes one price observation. `advance(...)` updates
state without returning a Python object. Multi-output `batch(...)` returns
arrays for `value`, `position`, and `width`.

## Theory Of Operation

The oscillator reuses the same recursive SMA stack as `RainbowMovingAverage`.
When the rainbow is wide relative to price, trend dispersion across lag depths
is large; when it is narrow, the layers have converged. RTTA also reports where
price sits between the highest and lowest rainbow layers, which is useful as a
normalized location feature.

## Recurrence

Let \(x_t\) be price and let \(H_t\), \(L_t\), \(\operatorname{width}_t = H_t-L_t\),
and \(\operatorname{mid}_t = \tfrac12(H_t+L_t)\) be the rainbow envelope from
`RainbowMovingAverage` with the same `period` and `layers`.

\[
\operatorname{value}_t = 100\cdot\frac{\operatorname{width}_t}{x_t}
\]

\[
\operatorname{position}_t = 100\cdot\frac{x_t - \operatorname{mid}_t}{\operatorname{width}_t}
\]

Division by zero (flat rainbow or zero price) is handled by the library's safe
divide and returns \(0\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class RainbowOscillator`, which owns an internal `RainbowMovingAverage`.

## Reference

- [Mel Widner, "Rainbow Charts," *Technical Analysis of Stocks & Commodities*,
  July 1997 (PDF mirror)](https://c.mql5.com/forextsd/forum/64/rainbow_oscillator_-_original_article_-_mel_widner.pdf)
- [Quantified Strategies: Rainbow Oscillator](https://www.quantifiedstrategies.com/rainbow-oscillator/)
