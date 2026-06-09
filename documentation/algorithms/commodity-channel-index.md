# CommodityChannelIndex

## Summary

`CommodityChannelIndex` is RTTA's streaming implementation of: CCI deviation of typical price from its moving average.

## Update API

```python
result = rtta.CommodityChannelIndex().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`CommodityChannelIndex` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CommodityChannelIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/commodity-channel-index-cci)
