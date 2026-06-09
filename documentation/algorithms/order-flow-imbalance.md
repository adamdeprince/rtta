# OrderFlowImbalance

## Summary

`OrderFlowImbalance` is RTTA's streaming implementation of: Quote-level best bid/ask price and size change pressure over a rolling update window.

## Update API

```python
result = rtta.OrderFlowImbalance().update(bid_price, bid_size, ask_price, ask_size)
```

The `update(...)` call consumes one observation using `bid_price`, `bid_size`, `ask_price`, `ask_size`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`OrderFlowImbalance` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = (bid_price_t, bid_size_t, ask_price_t, ask_size_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
W_t = \operatorname{push}(W_{t-1}, z_t, n)
\]

\[
y_t = G(W_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class OrderFlowImbalance`.

## Reference

- [Background reference](https://arxiv.org/abs/1011.6402)
