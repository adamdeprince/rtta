# HistoricalVolatility

## Summary

`HistoricalVolatility` is RTTA's streaming implementation of: Annualized rolling standard deviation of log returns.

## Update API

```python
result = rtta.HistoricalVolatility().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`HistoricalVolatility` estimates realized volatility from close-to-close log
returns over a rolling window and annualizes with
\(\sqrt{\texttt{periods\_per\_year}}\) (default 252).

## Recurrence

Let \(c_t = close_t\), \(n\) the window, and \(P\) periods per year.

\[
r_t = \ln\frac{c_t}{c_{t-1}}
\]

\[
\sigma_t = \operatorname{stddev}_n(r_t), \qquad
HV_t = \sigma_t \sqrt{P}
\]

`stddev` uses the same population form as [`StdDev`](std-dev.md)
(\(1/n\) second moment). The first sample has no return and yields 0 when
`fillna=True`, else NaN.

The return value is the current scalar indicator value.

## Composed Primitives

[`StdDev`](std-dev.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class HistoricalVolatility`.

## Reference

- [Background reference](https://www.investopedia.com/terms/h/historicalvolatility.asp)
