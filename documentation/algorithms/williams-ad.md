# WilliamsAD

## Summary

`WilliamsAD` is RTTA's streaming Williams Accumulation/Distribution line. It
cumulates a close-based accumulation amount on up closes and a distribution
amount on down closes, using high/low when they extend beyond the previous
close.

## Update API

```python
value = rtta.WilliamsAD().update(high, low, close)
```

The first bar seeds previous close and returns `0.0`. Unchanged closes add
nothing.

## Theory Of Operation

Larry Williams' A/D (distinct from the Chaikin Accumulation/Distribution line)
adds the distance from the true low of the move into the close on up days, and
subtracts the distance from the true high of the move into the close on down
days. Rising Williams A/D with rising prices confirms accumulation; divergence
can warn of weakening participation.

## Recurrence

Let \(H_t, L_t, C_t\) be high, low, close. Seed \(WAD_0 = 0\) and store \(C_0\).
For \(t \ge 1\):

\[
WAD_t =
\begin{cases}
WAD_{t-1} + \bigl(C_t - \min(L_t, C_{t-1})\bigr), & C_t > C_{t-1} \\[4pt]
WAD_{t-1} + \bigl(C_t - \max(H_t, C_{t-1})\bigr), & C_t < C_{t-1} \\[4pt]
WAD_{t-1}, & C_t = C_{t-1}
\end{cases}
\]

Then set previous close to \(C_t\). Note that on down closes the added term is
negative or zero, so the line declines.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class WilliamsAD`.

## Reference

- [Investopedia: Accumulation/Distribution (Williams form context)](https://www.investopedia.com/terms/a/accumulationdistribution.asp)
- [TradingPedia: Williams Accumulation/Distribution](https://www.tradingpedia.com/forex-trading-indicators/williams-accumulation-distribution/)
