# OnlineMarkovSwitchingVolatilityFilter

## Summary

`OnlineMarkovSwitchingVolatilityFilter` is RTTA's streaming implementation of: Online two-state Markov-switching volatility filter over close-to-close moves.

## Update API

```python
result = rtta.OnlineMarkovSwitchingVolatilityFilter().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`OnlineMarkovSwitchingVolatilityFilter` maintains online probabilities for latent states or components. An update combines the previous probabilities with the new observation likelihoods and normalizes the result.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\tilde{\pi}_t = A^\top \pi_{t-1}
\]

\[
\pi_t(i)=
\frac{\tilde{\pi}_t(i)\,p(z_t\mid i)}
{\sum_j \tilde{\pi}_t(j)\,p(z_t\mid j)}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class OnlineMarkovSwitchingVolatilityFilter`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Markov-switching_model)
