# CDLMatchingLow

## Summary

`CDLMatchingLow` is RTTA's streaming detector for the **Matching low** candlestick pattern.
Two consecutive bearish bars with nearly equal closes (possible support).

Output follows the TA-Lib convention: **`+100`** for a bullish match, **`-100`**
for a bearish match, and **`0`** when the pattern does not fire.

## Update API

```python
value = rtta.CDLMatchingLow(fillna=True).update(open, high, low, close)
# +100 bullish match, -100 bearish match (when directional), 0 no match
batch = rtta.CDLMatchingLow(fillna=True).batch(open, high, low, close)
```

`update(...)` consumes one OHLC bar. `advance(...)` uses the same inputs without
returning a Python value. Scalar `batch(open, high, low, close)` matches sequential
`update` on a fresh instance. With `fillna=False`, values are `NaN` until at least
2 bar(s) have been seen; with `fillna=True` (default), unmatched bars return `0`.


## Theory Of Operation

Candlestick patterns are short causal labels on bar geometry. They do not
forecast returns by themselves; they tag structure (indecision, rejection,
engulfing pressure, multi-bar reversals) that you can combine with trend,
volatility, or volume context.




On every bar the engine forms:

\[
\begin{aligned}
\mathrm{body}_t &= |C_t - O_t| \\
\mathrm{range}_t &= H_t - L_t \\
\mathrm{upper}_t &= H_t - \max(O_t, C_t) \\
\mathrm{lower}_t &= \min(O_t, C_t) - L_t \\
\mathrm{top}_t &= \max(O_t, C_t),\quad
\mathrm{bot}_t = \min(O_t, C_t) \\
\mathrm{mid}_t &= \tfrac12(O_t + C_t)
\end{aligned}
\]

A bar is **bullish** when \(C_t \ge O_t\) and **bearish** when \(C_t < O_t\).
Bars are stored in a short ring; age \(0\) is the newest bar, age \(1\) the prior
bar, and so on.

Shared predicates used below:

\[
\begin{aligned}
\mathrm{doji}(b) &\iff \mathrm{body} \le 0.1\cdot \mathrm{range}
  \quad(\text{or range}=0) \\
\mathrm{longBody}(b,\overline B,f) &\iff \mathrm{body} \ge f\cdot \overline B \\
\mathrm{shortBody}(b,\overline B,f) &\iff \mathrm{body} \le f\cdot \overline B \\
\mathrm{longLower}(b,m) &\iff \mathrm{lower} \ge m\cdot \mathrm{body} \\
\mathrm{longUpper}(b,m) &\iff \mathrm{upper} \ge m\cdot \mathrm{body} \\
\mathrm{smallUpper}(b,f) &\iff \mathrm{upper} \le f\cdot \mathrm{range} \\
\mathrm{smallLower}(b,f) &\iff \mathrm{lower} \le f\cdot \mathrm{range} \\
\mathrm{engulf}(c,p) &\iff
  \mathrm{top}_c \ge \mathrm{top}_p \land
  \mathrm{bot}_c \le \mathrm{bot}_p \land
  \mathrm{body}_c > \mathrm{body}_p \\
\mathrm{inside}(c,p) &\iff
  \mathrm{top}_c \le \mathrm{top}_p \land
  \mathrm{bot}_c \ge \mathrm{bot}_p
\end{aligned}
\]

\(\overline B\) is the average body over the last up to five ring bars;
\(\overline R\) is the average range over the same window.

For hammer-family patterns, a light **prior trend** uses the previous close
versus the SMA of the two closes before it:

\[
\tau =
\begin{cases}
+1 & C_{-1} > 1.0001\cdot \mathrm{SMA}(C_{-1},C_{-2}) \\
-1 & C_{-1} < 0.9999\cdot \mathrm{SMA}(C_{-1},C_{-2}) \\
0 & \text{otherwise}
\end{cases}
\]


## Recurrence

Push \((O_t,H_t,L_t,C_t)\) into the ring. Detection runs only when the ring has
at least **2** bar(s).


Emit \(+100\) when both bars are bearish and
\[
|C_c - C_p| \le 0.1\cdot \overline R.
\]


If the predicate fails, emit \(0\). The C++ path is \(O(1)\) per update (fixed
ring, no full-window rescan beyond a few bars).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CDLMatchingLow`,
built on shared `CdlHistory` / `cdl::` geometry helpers. Thresholds are
streaming-friendly geometric rules; they are **not** a line-by-line port of every
TA-Lib average-body lookback table.

## Reference

- [ChartSchool pattern dictionary](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary)
- [ChartSchool: Candlestick pattern dictionary](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary)
- [ChartSchool: Introduction to candlesticks](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/introduction-to-candlesticks)
