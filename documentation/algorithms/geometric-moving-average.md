# GeometricMovingAverage

## Summary

`GeometricMovingAverage` is RTTA's streaming geometric mean of recent prices:
the exponential of an SMA of log-prices. Prices must be strictly positive and
finite; invalid samples return `NaN` without advancing a usable log mean.

## Update API

```python
value = rtta.GeometricMovingAverage(window=14, fillna=True).update(price)
```

The nested log-SMA inherits `fillna`. Non-positive or non-finite `price` values
return `NaN` for that call.

## Theory Of Operation

The geometric mean of positive prices \(x_1,\ldots,x_n\) is

\[
\Bigl(\prod_{i=1}^{n} x_i\Bigr)^{1/n} = \exp\!\Bigl(\tfrac1n\sum_{i=1}^{n}\log x_i\Bigr).
\]

This is the natural smoother for multiplicative (return-like) processes: equal
percentage moves affect the geometric mean symmetrically, whereas an arithmetic
SMA weights absolute points equally. RTTA implements the identity
\(\operatorname{GMA} = \exp(\operatorname{SMA}(\log price))\).

## Recurrence

Let \(x_t\) be the input price and \(n\) be `window` (default \(14\)).

If \(x_t \le 0\) or \(x_t\) is not finite, return `NaN`. Otherwise:

\[
\ell_t = \log x_t, \qquad
L_t = \operatorname{SMA}_n(\ell_t)
\]

\[
GMA_t = \exp(L_t)
\]

If the nested SMA returns `NaN` (warmup with `fillna=False`), the output is
`NaN`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class GeometricMovingAverage`. The member `logs_` is an `SMA` fed with
`std::log(price)`.

## Reference

- [Investopedia: Geometric Mean](https://www.investopedia.com/terms/g/geometricmean.asp)
- [Wikipedia: Geometric mean](https://en.wikipedia.org/wiki/Geometric_mean)
