# OnlineGaussianMixtureRegimeFilter

## Summary

`OnlineGaussianMixtureRegimeFilter` is RTTA's streaming implementation of: Online Gaussian mixture regime filter with bounded component count.

## Update API

```python
result = rtta.OnlineGaussianMixtureRegimeFilter().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`OnlineGaussianMixtureRegimeFilter` converts each observation into a streaming score and then applies threshold or hysteresis logic. The state is deliberately sticky where the C++ class models regimes, so small reversals do not immediately flip the output.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
r_{t,k}= \frac{w_{t-1,k}p(z_t\mid k)}
{\sum_j w_{t-1,j}p(z_t\mid j)}
\]

\[
w_{t,k}=(1-\alpha)w_{t-1,k}+\alpha r_{t,k}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class OnlineGaussianMixtureRegimeFilter`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Mixture_model)
