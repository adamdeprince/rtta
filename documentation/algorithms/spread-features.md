# SpreadFeatures

## Summary

`SpreadFeatures` is RTTA's streaming implementation of: Quoted, effective, and realized spread estimates from trades and contemporaneous quotes.

## Update API

```python
result = rtta.SpreadFeatures().update(trade_price, bid_price, ask_price)
```

The `update(...)` call consumes one observation using `trade_price`, `bid_price`, `ask_price`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SpreadFeatures` combines price, volume, and/or quote information into a streaming microstructure or participation measure. The update path advances only from the latest tick and prior state.

## Recurrence

Let \(z_t = (trade_price_t, bid_price_t, ask_price_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
mid_t=\frac{bid_t+ask_t}{2}, \qquad
spread_t=\max(ask_t-bid_t,0)
\]

\[
relative\_spread_t=\frac{spread_t}{\max(|mid_t|,\epsilon)}, \qquad
trade\_location_t=\frac{trade_t-mid_t}{\max(spread_t,\epsilon)}
\]

`update(...)` returns a result struct with fields `quoted_spread`, `effective_spread`, `realized_spread`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SpreadFeatures`.

## Reference

- [Background reference](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
