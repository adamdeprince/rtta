# DecomposedOrderFlowImbalance

## Summary

`DecomposedOrderFlowImbalance` splits Cont-style best-quote pressure into three additive
channels — **add**, **cancel**, and **trade** (proxy) — each rolled with a sum window, plus
their sum as `total`. Inputs are top-of-book bid/ask price and size only.

## Update API

```python
import rtta

ind = rtta.DecomposedOrderFlowImbalance(window=1, fillna=True)
result = ind.update(bid_price, bid_size, ask_price, ask_size)
# result.add, result.cancel, result.trade, result.total
```

`advance(...)` updates state without returning a result. Batch helpers accept parallel
arrays of bid/ask price and size.

## Theory Of Operation

Standard Cont OFI mixes liquidity additions, cancellations, and aggressive trades into one
scalar. This indicator attributes each quote revision to:

- **Add** — new liquidity that increases Cont-style buy pressure (bid improve / size
  increase, or ask worsen in the Cont sign convention).
- **Cancel** — removed liquidity with opposite Cont signs.
- **Trade** — heuristic: mid moves up while ask size decreases without an ask price
  improvement (buyer lifting offers), or mid moves down while bid size decreases without
  bid price worsening (seller hitting bids).

Each component is accumulated in its own `RollingSumWindow`. The first snapshot seeds
previous state with zero contributions.

## Recurrence

Let \((b_t, B_t, a_t, A_t)\) be top-of-book bid/ask price and size, and
\(m_t = \tfrac12(b_t + a_t)\). On the first tick, instantaneous contributions are zero.
Thereafter, instantaneous pieces \(e^{\mathrm{add}}_t\), \(e^{\mathrm{cancel}}_t\),
\(e^{\mathrm{trade}}_t\) are formed as follows.

**Bid side**

\[
\begin{aligned}
b_t > b_{t-1} &\Rightarrow e^{\mathrm{add}} {+}{=} B_t,\\
b_t < b_{t-1} &\Rightarrow e^{\mathrm{cancel}} {-}{=} B_{t-1},\\
b_t = b_{t-1},\; B_t > B_{t-1} &\Rightarrow e^{\mathrm{add}} {+}{=} B_t - B_{t-1},\\
b_t = b_{t-1},\; B_t < B_{t-1} &\Rightarrow e^{\mathrm{cancel}} {-}{=} B_{t-1} - B_t.
\end{aligned}
\]

**Ask side** (Cont signs flipped relative to bid)

\[
\begin{aligned}
a_t < a_{t-1} &\Rightarrow e^{\mathrm{add}} {-}{=} A_t,\\
a_t > a_{t-1} &\Rightarrow e^{\mathrm{cancel}} {+}{=} A_{t-1},\\
a_t = a_{t-1},\; A_t > A_{t-1} &\Rightarrow e^{\mathrm{add}} {-}{=} A_t - A_{t-1},\\
a_t = a_{t-1},\; A_t < A_{t-1} &\Rightarrow e^{\mathrm{cancel}} {+}{=} A_{t-1} - A_t.
\end{aligned}
\]

**Trade proxy**

\[
\begin{aligned}
m_t > m_{t-1} \;\land\; A_t < A_{t-1} \;\land\; a_t \le a_{t-1}
&\Rightarrow e^{\mathrm{trade}} {+}{=} A_{t-1} - A_t,\\
m_t < m_{t-1} \;\land\; B_t < B_{t-1} \;\land\; b_t \ge b_{t-1}
&\Rightarrow e^{\mathrm{trade}} {-}{=} B_{t-1} - B_t.
\end{aligned}
\]

Rolling outputs (\(W =\) `window`):

\[
\mathrm{add}_t = \sum e^{\mathrm{add}},\quad
\mathrm{cancel}_t = \sum e^{\mathrm{cancel}},\quad
\mathrm{trade}_t = \sum e^{\mathrm{trade}}
\quad\text{(last \(W\) samples)},
\]

\[
\mathrm{total}_t = \mathrm{add}_t + \mathrm{cancel}_t + \mathrm{trade}_t.
\]

If `fillna=False` and any window is incomplete, all four fields are NaN.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class DecomposedOrderFlowImbalance`. Three independent `RollingSumWindow` instances hold
add/cancel/trade.

## Reference

- [Cont, Kukanov & Stoikov, “The Price Impact of Order Book Events” (arXiv:1011.6402)](https://arxiv.org/abs/1011.6402)
