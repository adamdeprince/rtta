# Ichimoku

## Summary

`Ichimoku` is a streaming Ichimoku Kinko Hyo component set: conversion line (Tenkan-sen),
base line (Kijun-sen), leading Span A and Span B (undelayed causal values), Chikou /
lagging span, and **cloud spans delayed by `window2`** so they align with the bar where
the cloud is traditionally plotted (`span_a_displaced`, `span_b_displaced`).

## Update API

```python
import rtta

ind = rtta.Ichimoku(window1=9, window2=26, window3=52, fillna=True)
result = ind.update(high, low, close)
# result.conversion, result.base, result.span_a, result.span_b,
# result.lagging_span,
# result.span_a_displaced, result.span_b_displaced
```

`advance(...)` updates state without returning a result. Batch returns parallel arrays
for all seven fields.

## Theory Of Operation

Ichimoku uses midpoints of rolling high–low ranges:

| Component | Classic name | Definition |
|-----------|--------------|------------|
| conversion | Tenkan-sen | midpoint of high/low over `window1` |
| base | Kijun-sen | midpoint over `window2` |
| span_a | Senkou Span A (raw) | midpoint of conversion and base |
| span_b | Senkou Span B (raw) | midpoint of high/low over `window3` |
| lagging_span | Chikou Span | close delayed by `window2` |
| span_a_displaced | cloud A at “now” | raw span_a delayed by `window2` |
| span_b_displaced | cloud B at “now” | raw span_b delayed by `window2` |

In classical charts, Senkou spans are plotted **forward** by 26 bars. In a causal stream
you cannot emit future values, so RTTA also emits the spans that **arrive** on the
current bar: the values computed `window2` steps ago. The undelayed `span_a` / `span_b`
remain available for custom leading-cloud logic.

With `fillna=False`, incomplete range windows yield NaN for that component. Delay lines
use the same `fillna` policy; NaN raw spans are replaced by `close` when filling so the
delay buffer stays defined.

## Recurrence

Let \(n_1,n_2,n_3\) be `window1`, `window2`, `window3`. Rolling extremes over the last
\(n\) highs/lows:

\[
\begin{aligned}
C_t &= \tfrac12\bigl(\max_{i\in[t-n_1+1,t]} H_i + \min_{i\in[t-n_1+1,t]} L_i\bigr),\\
B_t &= \tfrac12\bigl(\max_{i\in[t-n_2+1,t]} H_i + \min_{i\in[t-n_2+1,t]} L_i\bigr),\\
A_t &= \tfrac12(C_t + B_t),\\
S_t &= \tfrac12\bigl(\max_{i\in[t-n_3+1,t]} H_i + \min_{i\in[t-n_3+1,t]} L_i\bigr).
\end{aligned}
\]

Lagging (Chikou) and displaced cloud (delays of length \(n_2\)):

\[
\mathrm{lagging\_span}_t = c_{t-n_2}
\quad\text{(Delay of close; NaN until the delay is full if `fillna=False`)},
\]

\[
\begin{aligned}
A^{\mathrm{in}}_t &=
\begin{cases}
A_t, & A_t \text{ finite},\\
c_t, & \text{NaN and `fillna`}\\
\text{NaN}, & \text{NaN and not `fillna`},
\end{cases}
\qquad
\mathrm{span\_a\_displaced}_t = A^{\mathrm{in}}_{t-n_2},
\end{aligned}
\]

and likewise for \(S_t \rightarrow \mathrm{span\_b\_displaced}_t\).

Public fields: `conversion`\( = C_t\), `base`\( = B_t\), `span_a`\( = A_t\),
`span_b`\( = S_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Ichimoku` using
`RollingExtreme` for highs/lows and `Delay` for lagging and displaced spans. Result type
is `IchimokuResult` / `IchimokuBatchResult`.

## Reference

- [Investopedia — Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)
