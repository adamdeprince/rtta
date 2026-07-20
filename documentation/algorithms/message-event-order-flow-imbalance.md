# MessageEventOrderFlowImbalance

## Summary

`MessageEventOrderFlowImbalance` is RTTA's streaming message-tape order-flow
imbalance: a rolling sum of signed contributions from discrete LOB/trade events
(add, cancel, trade) rather than Cont-style best-quote snapshot deltas.

## Update API

```python
result = rtta.MessageEventOrderFlowImbalance(window=50, fillna=True).update(
    event_type, side, qty
)
# result.ofi, result.event, result.signed_size
```

- `event_type`: `1` = add, `2` = cancel, `3` = trade (other codes fall back to
  signed size only)
- `side`: `+1` = bid / buy aggressor, `-1` = ask / sell aggressor
- `qty`: non-negative event size (absolute value is used if negative)

`advance(...)` uses the same three inputs without returning a Python object.
Multi-output `batch(event_type, side, qty)` returns arrays for `ofi`, `event`,
and `signed_size`.

## Theory Of Operation

Snapshot OFI (`OrderFlowImbalance`) reconstructs pressure from changes in
best bid/ask price and size. Message-tape OFI instead consumes the exchange
event stream: liquidity adds, cancels, and trades. RTTA maps each event to a
signed contribution and accumulates those contributions over a rolling window of
events. Positive rolling sum means net bid/buy pressure in the recent message
history; negative means net ask/sell pressure.

## Recurrence

Let \(e_t\in\{1,2,3\}\) be the event type, \(s_t\in\{+1,-1\}\) the side, and
\(q_t\ge 0\) the quantity. Define the contribution

\[
\delta_t =
\begin{cases}
+s_t q_t & e_t = 1 \quad\text{(add)} \\
-s_t q_t & e_t = 2 \quad\text{(cancel)} \\
+s_t q_t & e_t = 3 \quad\text{(trade)} \\
+s_t q_t & \text{otherwise}
\end{cases}
\]

Over a rolling window \(W_t\) of the last \(n\) contributions (`window`):

\[
\operatorname{OFI}_t = \sum_{\tau\in W_t} \delta_\tau
\]

Each update also emits the latest contribution as `event` and the signed size
\(s_t q_t\) as `signed_size`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class MessageEventOrderFlowImbalance` using a `RollingSumWindow`. With
`fillna=False`, `ofi` is `NaN` until the window is full.

This is a research-facing event API; it is not interchangeable with
top-of-book `OrderFlowImbalance` without converting the feed.

## Reference

- [Cont, Kukanov, and Stoikov, "The Price Impact of Order Book Events,"
  arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
  ( Cont-style OFI motivation; RTTA's snapshot form is `OrderFlowImbalance`,
  while this class applies the same imbalance idea to explicit message events. )
- [Cont, Cucuringu, et al. multi-level / LOB flow literature
  overview](https://arxiv.org/abs/2104.14067)
