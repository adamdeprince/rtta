# Bias

## Summary

`Bias` is RTTA's streaming percentage deviation of price from its simple moving
average. It measures how far the current close sits above or below the SMA, as
a percent of that average.

## Update API

```python
value = rtta.Bias(window=20, fillna=True).update(close)
```

The `update(...)` call consumes one close. With `fillna=False`, output is `NaN`
until the SMA window is full (same warmup policy as the nested SMA).

## Theory Of Operation

Bias (common in Asian market practice, also called "percentage bias" or
"price bias ratio") is a normalized distance from a moving average. Positive
values mean price is extended above the mean; negative values mean it is below.
Because the denominator is the average itself, Bias is scale-free and comparable
across instruments.

RTTA implements Bias as \(100 \times (close - SMA) / SMA\) using a nested SMA.

## Recurrence

Let \(c_t\) be close and \(n\) be `window` (default \(20\)).

\[
S_t = \operatorname{SMA}_n(c_t)
\]

\[
\operatorname{Bias}_t = 100 \cdot \frac{c_t - S_t}{S_t}
\quad\text{(safe divide)}
\]

When \(S_t = 0\) or is non-finite, the safe-divide path returns `0` / `NaN`
consistent with RTTA's `safe_divide` helper.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Bias`. The
SMA member is constructed with the same `fillna` flag as the outer indicator.

## Reference

- [Investopedia: Moving Average](https://www.investopedia.com/terms/m/movingaverage.asp)
- [TradingView: Bias indicator concept](https://www.tradingview.com/script/kI9g2u0a-Bias/)
