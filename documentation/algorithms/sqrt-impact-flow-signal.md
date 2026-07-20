# SqrtImpactFlowSignal

## Summary

`SqrtImpactFlowSignal` is a streaming research signal built from the **square-root
market impact** law and Cont-style signed order flow. It is designed for
Massive/Polygon **aggregate bars** (close, volume, optional VWAP) and can also
consume **tick-derived signed dollar volume** when you aggregate the tape yourself.

The economic idea:

1. Average price impact of a metaorder scales like \(Y\,\sigma\sqrt{Q/V}\)
   (Tóth et al.; Bouchaud surveys; 2025 work on impact, imbalance, and volatility).
2. If a bar’s signed volume implies more impact than the return that realized,
   impact is “unfinished” → **continuation** in the flow direction.
3. If the return overshoots volume-implied impact → **temporary impact reversion**.
4. Optional **VWAP gap** aligns trade-price pressure with the bar’s volume-weighted
   location (Polygon aggregate `vw`).

## Update API

```python
# Aggregate path (tick-rule flow from close-to-close)
out = rtta.SqrtImpactFlowSignal().update(close, volume)
# out.signal, out.score, out.impact, out.residual, out.continuation,
# out.reversion, out.participation, out.flow, out.volatility, out.vwap_gap

# Optional: true signed dollar volume from ticks + bar VWAP
out = rtta.SqrtImpactFlowSignal().update(
    close, volume, signed_dollar_volume=signed_dv, vwap=bar_vwap
)

# OHLCV-compatible overload (open/high/low ignored)
out = rtta.SqrtImpactFlowSignal().update(open, high, low, close, volume)

batch = rtta.SqrtImpactFlowSignal().batch(close, volume)
```

Constructor knobs:

| Argument | Default | Role |
|----------|---------|------|
| `impact_coefficient` | `1.0` | \(Y\) in \(I = Y\sigma\sqrt{Q/V}\) |
| `adv_span` | `50` | EWMA span for average dollar volume \(V\) |
| `vol_span` | `20` | EWMA span for return scale \(\sigma\) |
| `continuation_weight` | `1.0` | weight on unfinished impact |
| `reversion_weight` | `0.5` | weight on overshoot reversion |
| `vwap_weight` | `0.25` | weight on VWAP-gap alignment |
| `entry_z` / `exit_z` | `0.75` / `0.25` | hysteresis for discrete `signal` |
| `fillna` | `True` | `0` vs `NaN` on warmup |

## Theory Of Operation

### Square-root impact

Empirically and theoretically, the average absolute mid/price displacement
associated with trading quantity \(Q\) against recent activity \(V\) is

\[
I_t = Y\,\sigma_t\sqrt{\frac{Q_t}{V_t}}
\]

with \(\sigma_t\) a local volatility scale. RTTA sets \(Q_t = C_t V^{\mathrm{sh}}_t\)
(dollar volume of the bar) and \(V_t\) an EWMA of dollar volume (ADV proxy).

### Signed flow

On aggregates without a tape, flow sign is the **tick rule** on close:

\[
s_t = \mathrm{sign}(\log C_t - \log C_{t-1}).
\]

When you supply `signed_dollar_volume` from tick classification (tick rule /
Lee-Ready on quotes), \(s_t = \mathrm{sign}(\mathrm{signed\_dollar})\).

### Continuation vs reversion

Let \(r_t = \log(C_t/C_{t-1})\) and \(I_t\) as above.

\[
\begin{aligned}
\mathrm{continuation}_t &= s_t\cdot\frac{\max(0,\, I_t - |r_t|)}{\sigma_t} \\
\mathrm{reversion}_t &= -\mathrm{sign}(r_t)\cdot\frac{\max(0,\, |r_t| - I_t)}{\sigma_t}
\end{aligned}
\]

- **Continuation** > 0: heavy signed volume but a small move → expect more drift
  with the flow (incomplete permanent impact / underreaction).
- **Reversion** > 0: move larger than volume-implied impact → temporary impact
  should mean-revert.

### VWAP alignment

If bar VWAP \(W_t\) is available:

\[
g_t = \frac{C_t}{W_t} - 1, \qquad
\mathrm{align}_t = s_t\cdot\frac{g_t}{\sigma_t}.
\]

Buying into a close above VWAP (or selling below) reinforces pressure.

### Score and discrete signal

\[
\mathrm{raw}_t = \tanh\bigl(
  w_c\,\mathrm{continuation}_t + w_r\,\mathrm{reversion}_t + w_v\,\mathrm{align}_t
\bigr)
\]

`score` is \(\mathrm{raw}_t\). An online z-score of `score` with hysteresis
(`entry_z` / `exit_z`) produces discrete `signal` \(\in \{-1,0,+1\}\).

## Recurrence

State: previous close, EWMA ADV \(V_t\), EWMA \(|r|\) as \(\sigma_t\), score EWMA
mean/variance, discrete position.

1. \(r_t = \log(C_t/C_{t-1})\), \(Q_t = C_t\cdot\mathrm{volume}_t\).
2. \(V_t = (1-\alpha_V)V_{t-1} + \alpha_V Q_t\).
3. \(\sigma_t = (1-\alpha_\sigma)\sigma_{t-1} + \alpha_\sigma |r_t|\).
4. \(I_t = Y\sigma_t\sqrt{Q_t/V_t}\), residual \(r_t - s_t I_t\).
5. Form continuation, reversion, VWAP align → tanh score → z-hysteresis signal.

All updates are \(O(1)\) and causal.

## Trading (buy / sell)

The discrete trade directive is **`result.signal`**:

| `signal` | Action (see example) |
|----------|----------------------|
| `+1` | **BUY** / hold long |
| `-1` | **SELL** short, or exit long if long-only |
| `0` | **FLAT** / exit |

`score` is optional confidence/size. A full paper-trading loop (CSV with
`action` in `{BUY,SELL,COVER,HOLD,...}`, long-only vs `--allow-short`,
`--min-score`) lives in:

`examples/sqrt_impact_flow_from_massive_speedup.py`

## Implementation Notes

Implemented in `src/rtta/indicator.cpp` in `class SqrtImpactFlowSignal`.

This is **not** a full LOB metaorder reconstructor. It is a bar-level / optional
tape-signed proxy of the square-root impact residual structure emphasized in
recent impact–imbalance research, engineered for Polygon/Massive fields.

## Reference

- Tóth et al., “Anomalous price impact and the critical nature of liquidity in
  limit order books,” *Physical Review X*, 2011 (square-root impact phenomenology).
- Bouchaud, Farmer, Lillo — surveys on market impact and the square-root law.
- [arXiv:2509.05065](https://arxiv.org/abs/2509.05065) — *The Subtle Interplay
  between Square-root Impact, Order Imbalance & Volatility II* (2025): correlation
  structure between generalized order flow and returns.
- Cont, Kukanov, Stoikov, “The Price Impact of Order Book Events,”
  [arXiv:1011.6402](https://arxiv.org/abs/1011.6402) (OFI / signed flow backbone).
