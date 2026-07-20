# QStick

## Summary

`QStick` is RTTA's streaming QStick indicator: a simple moving average of the
close-minus-open candle body. Positive values mean recent candles have been
mostly bullish; negative values mean mostly bearish.

## Update API

```python
value = rtta.QStick(window=14, fillna=True).update(open, close)
```

The nested SMA inherits `fillna`.

## Theory Of Operation

Tushar Chande's QStick summarizes the average signed body of recent candles.
Unlike pure close-based momentum, it uses the open-to-close move of each bar, so
large-range bars that close near the open contribute little. Zero crossings can
be treated as short-term trend shifts; extremes can flag exhaustion.

## Recurrence

Let \(O_t, C_t\) be open and close and \(n\) be `window` (default \(14\)).

\[
b_t = C_t - O_t
\]

\[
QStick_t = \operatorname{SMA}_n(b_t)
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class QStick`.
The sole member is an `SMA` fed with `close - open`.

## Reference

- [Investopedia: Qstick Indicator](https://www.investopedia.com/terms/q/qstick.asp)
- [TradingPedia: QStick](https://www.tradingpedia.com/forex-trading-indicators/qstick-indicator/)
