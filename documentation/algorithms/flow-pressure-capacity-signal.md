# FlowPressureCapacitySignal

## Summary

`FlowPressureCapacitySignal` is a streaming, event-time L1 microstructure signal.
It measures whether recent buyer- or seller-initiated trade flow is large relative
to the displayed liquidity standing in its way, then asks whether that liquidity
was **depleted, replenished, or withdrawn**.

That distinction is the idea. Raw order-book imbalance cannot tell a genuine
liquidity vacuum from a large queue that flickers for one update. Raw signed flow
cannot tell an ask being consumed from an ask that refills every time it is hit.
The signal combines both tapes causally:

1. event-time filtered L1 queue imbalance;
2. aggressive buy/sell flow divided by opposing near-touch capacity;
3. inferred same-price replenishment after executions; and
4. bid/ask withdrawal and price-level depletion as liquidity fragility.

The implementation is an RTTA engineering synthesis, not a verbatim estimator
from one paper. Its main empirical motivation is Chang (2026), which finds that
directional flow relative to near-touch capacity is more informative than raw
flow for short-horizon return and adverse-selection risk. It also incorporates
the real-time structural-filtration result of Nittur Anantha, Jain, and Maiti
(2025): transient book activity weakens raw imbalance, while trade-event
imbalance has stronger causal alignment with future price movement.

## Update API

The compact API accepts net signed aggressive volume:

```python
indicator = rtta.FlowPressureCapacitySignal()

out = indicator.update(
    bid_price,
    bid_size,
    ask_price,
    ask_size,
    signed_trade_volume,  # buys positive, sells negative
)
```

When buys and sells both occur between quote updates, the six-input overload is
lossless and preferred:

```python
out = indicator.update(
    bid_price,
    bid_size,
    ask_price,
    ask_size,
    buy_volume,
    sell_volume,
)
```

`buy_volume` and `sell_volume` are aggressive volumes observed **after the
previous accepted quote and no later than the current quote**. Quote sizes and
trade volumes must use the same unit.

The usual RTTA state APIs are available:

```python
indicator.advance(bid, bid_size, ask, ask_size, signed_volume)
out = indicator.last()
indicator.reset()

batch = indicator.batch(
    bid_prices, bid_sizes, ask_prices, ask_sizes, signed_volumes
)

detailed_batch = indicator.batch(
    bid_prices, bid_sizes, ask_prices, ask_sizes, buy_volumes, sell_volumes
)
```

NumPy `float32` and `float64` batches are supported. Pandas table batches use
columns `bid_price`, `bid_size`, `ask_price`, `ask_size`, and `signed_volume`.

## Constructor

| Argument | Default | Meaning |
|---|---:|---|
| `half_life_updates` | `32.0` | Event-count half-life for queue, flow, replenishment, and withdrawal state. |
| `queue_weight` | `0.25` | Weight on persistent displayed queue imbalance. |
| `pressure_weight` | `1.0` | Weight on aggressive flow relative to opposing capacity. |
| `replenishment_weight` | `0.75` | Weight on signed absorption: ask refills resist buys; bid refills resist sells. |
| `fragility_weight` | `0.25` | Weight on ask withdrawal minus bid withdrawal. |
| `score_scale` | `1.0` | Scale applied before final `tanh` saturation. |
| `entry_threshold` | `0.35` | Absolute score required to leave flat. |
| `exit_threshold` | `0.15` | Hysteresis threshold for returning to flat. |
| `warmup` | `8` | Valid quote updates required before `signal` can arm. |
| `fillna` | `True` | Return diagnostic values during warmup (`False` returns NaNs). |

## Outputs

| Field | Meaning |
|---|---|
| `signal` | Discrete stance in `{-1, 0, +1}` with entry/exit hysteresis. |
| `score` | Continuous composite pressure score in `[-1, 1]`. |
| `fair_value` | Midpoint plus one half-spread times `score`; always inside the current quote. |
| `microprice` | Standard size-weighted L1 microprice. |
| `raw_queue_imbalance` | Instantaneous `(bid_size - ask_size) / total_size`. |
| `queue_imbalance` | Event-time exponentially filtered queue imbalance. |
| `flow_imbalance` | Decayed aggressive `(buy - sell) / (buy + sell)` flow. |
| `pressure` | Signed log ratio of aggressive flow to opposing capacity. |
| `replenishment` | Signed absorption channel; positive is bid support, negative is ask supply. |
| `fragility` | Ask-side withdrawal minus bid-side withdrawal; positive is bullish. |
| `spread_bps` | Current quoted spread relative to midpoint in basis points. |

`fair_value` is a bounded diagnostic, not a claim that crossing the spread is
profitable. Validate `score` against a future midpoint and test the strategy at
delayed bid/ask prices, as the Massive example does.

## Theory of Operation

### Queue imbalance and microprice

For best bid/ask prices \(P^b_t,P^a_t\) and displayed sizes \(Q^b_t,Q^a_t\),

\[
I_t = \frac{Q^b_t-Q^a_t}{Q^b_t+Q^a_t}, \qquad
M_t = \frac{P^b_t+P^a_t}{2}.
\]

The familiar weighted mid is

\[
\operatorname{micro}_t =
\frac{P^a_t Q^b_t + P^b_t Q^a_t}{Q^b_t+Q^a_t}
= M_t + \frac{P^a_t-P^b_t}{2} I_t.
\]

Stoikov's microprice formalizes fair price as the expected limiting midpoint
conditional on spread and imbalance. Recent high-resolution work likewise starts
from this L1 displacement and corrects it using order-book dynamics. Here the
dynamic correction is deliberately small, transparent, and online.

To prevent one-update size flicker from dominating, RTTA filters imbalance in
event time. With half-life \(H\), \(d=2^{-1/H}\):

\[
\bar I_t = d\bar I_{t-1} + (1-d)I_t.
\]

### Execution-adjusted queue accounting

Suppose the ask price is unchanged between two quote snapshots and aggressive
buy volume \(V^+_t\) traded in between. Ignoring hidden liquidity, the visible
queue identity is

\[
Q^a_t = Q^a_{t-1} - V^+_t + A^a_t - C^a_t,
\]

where \(A\) is added liquidity and \(C\) is canceled liquidity. Only their net
is identifiable from L1, so the indicator records

\[
R^a_t=\max(Q^a_t-Q^a_{t-1}+V^+_t,0),\qquad
W^a_t=\max(Q^a_{t-1}-V^+_t-Q^a_t,0).
\]

The bid equations replace \(V^+\) with aggressive sell volume \(V^-\). When an
ask moves up or a bid moves down, the prior queue is treated as withdrawn or
depleted. An improving ask or bid is treated as new replenishment at the touch.

This is why correctly merging trades *between* consecutive quotes matters. If a
trade is attached to a future quote, the routine will mistake execution for
replenishment and leak information.

### Decayed flow and capacity

Aggressive flows, replenishment, and withdrawal use exponential shot-noise
state rather than a rectangular window:

\[
\bar X_t = d\bar X_{t-1}+X_t.
\]

Effective opposing capacities are

\[
C^a_t=Q^a_t+\bar R^a_t,\qquad C^b_t=Q^b_t+\bar R^b_t.
\]

The pressure channel compares buyer pressure against ask capacity and seller
pressure against bid capacity on a symmetric, bounded scale:

\[
P_t=\tanh\!\left[
\log\!\left(1+\frac{\bar V^+_t}{C^a_t}\right)
-\log\!\left(1+\frac{\bar V^-_t}{C^b_t}\right)
\right].
\]

A hundred shares buying into a ten-share ask therefore matters much more than
the same flow meeting ten thousand displayed shares.

### Absorption and fragility

Replenishment supplies the missing discriminator. Let

\[
A^a_t=\frac{\bar R^a_t}{\bar R^a_t+\bar V^+_t},\qquad
A^b_t=\frac{\bar R^b_t}{\bar R^b_t+\bar V^-_t}.
\]

Weighting by each side's share of recent aggressive flow gives

\[
R_t = w^-_t A^b_t - w^+_t A^a_t.
\]

Thus a bid repeatedly refilling against sells is bullish support; an ask
repeatedly refilling against buys is bearish supply. Full ask replenishment can
neutralize otherwise bullish buy pressure instead of blindly following it.

Withdrawal fragility is

\[
F_t =
\frac{\bar W^a_t}{\bar W^a_t+Q^a_t+\bar R^a_t}
-
\frac{\bar W^b_t}{\bar W^b_t+Q^b_t+\bar R^b_t}.
\]

An evaporating ask is bullish; an evaporating bid is bearish.

### Composite and signal

With constructor weights \(w_I,w_P,w_R,w_F\),

\[
S_t=\tanh\left\{s\left(
w_I\bar I_t+w_P P_t+w_R R_t+w_F F_t
\right)\right\}.
\]

`score` is \(S_t\), and

\[
\operatorname{fair}_t=M_t+\tfrac12(P^a_t-P^b_t)S_t.
\]

`signal` arms at `entry_threshold` and remains on that side until `score`
crosses `exit_threshold`, avoiding quote-by-quote position churn.

Every update is causal, allocation-free after construction, and \(O(1)\).

## Massive/Polygon Event Alignment

The reference example merges `StockQuoteDatabase` and `StockTradeDatabase` by
SIP timestamp. Each trade is classified against the last quote already known at
that timestamp. Buy and sell volumes are accumulated until the next quote, then
passed through the detailed overload. It also:

- skips locked/crossed/zero quotes;
- resets state at each regular session;
- evaluates sampled forecasts against a later midpoint;
- reports score/move correlation and signed edge; and
- paper-trades at delayed bid/ask prices so the spread is paid explicitly.

Quote and trade sizes must use the same unit. [Massive began reporting stock
quote sizes in shares rather than round lots on November 3,
2025](https://www.massive.com/changelog#stock-quote-size-reporting-change) and
later regenerated its historical quote flat files. Current files therefore use
the example's default `--quote-size-multiplier 1`; set it to `100` only for an
older cached file that still stores quote size in round lots.

See `examples/flow_pressure_capacity_from_massive_speedup.py`.

## Limitations

- Consolidated L1 size is not an order-ID queue. Additions and cancellations are
  net inferences, and hidden/iceberg liquidity is unobserved.
- SIP trade and quote feeds can have ambiguous same-timestamp ordering. Use
  participant timestamps only if both streams share a defensible clock/order.
- Trade conditions, corrections, odd lots, and off-exchange prints need a
  production eligibility policy. The example applies only minimal filtering.
- Event-time half-life adapts to activity but represents different wall-clock
  durations across names and regimes. Downsample first if clock-time comparability
  is required.
- The motivating pressure/capacity evidence includes crypto and non-US market
  data. The construction must be validated out of sample on the intended US
  equity universe before capital is committed.

## References

- Lawrence Chang, [“Do Order-Book States Predict Passive-Buy Toxicity? Evidence
  from BTC Perpetual Futures”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6693260),
  2026.
- Aditya Nittur Anantha, Shashi Jain, and Prithwish Maiti,
  [“Order Book Filtration and Directional Signal Extraction at High
  Frequency”](https://arxiv.org/abs/2507.22712), 2025.
- Christian D. Blakely, [“High Resolution Microprice Estimates from Limit
  Orderbook Data Using Hyperdimensional Vector Tsetlin
  Machines”](https://arxiv.org/abs/2411.13594), 2024.
- Sasha Stoikov, [“The Micro-Price: A High Frequency Estimator of Future
  Prices”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694), 2018.
- Rama Cont, Arseniy Kukanov, and Sasha Stoikov, [“The Price Impact of Order
  Book Events”](https://arxiv.org/abs/1011.6402), 2014.
- Yang Zhou, Jianwen Chen, and Ruipeng Wei, [“Order Splitting and Liquidity
  Replenishment Are Jointly Necessary for the Square-Root Law of Market
  Impact”](https://arxiv.org/abs/2607.04280), 2026.
