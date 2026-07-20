# MarketFacilitationIndex

## Summary

`MarketFacilitationIndex` is RTTA's per-bar Market Facilitation Index: the high
minus low range divided by volume. There is no rolling window or cumulative
state.

## Update API

```python
value = rtta.MarketFacilitationIndex().update(high, low, volume)
```

Each call is independent of prior bars. Zero volume yields a safe-divide result
of `0`.

## Theory Of Operation

Bill Williams' Market Facilitation Index (MFI, not to be confused with the Money
Flow Index) measures how much price range is "facilitated" per unit of volume.
A large range on light volume implies easy movement (high facilitation); a small
range on heavy volume implies heavy facilitation resistance. Williams often pairs
MFI with volume change to color-code bar types (green, fade, fake, squat); RTTA
returns only the scalar MFI itself.

## Recurrence

Let \(H_t, L_t, V_t\) be high, low, and volume.

\[
MFI_t = \frac{H_t - L_t}{V_t}
\quad\text{(safe divide)}
\]

No previous-bar state is retained.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class MarketFacilitationIndex`.

## Reference

- [Investopedia: Market Facilitation Index](https://www.investopedia.com/terms/m/marketfacilitationindex.asp)
- [TradingPedia: Market Facilitation Index](https://www.tradingpedia.com/forex-trading-indicators/market-facilitation-index/)
