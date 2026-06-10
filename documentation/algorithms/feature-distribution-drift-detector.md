# FeatureDistributionDriftDetector

## Summary

`FeatureDistributionDriftDetector` is RTTA's streaming implementation of: Bounded ADWIN-style drift detector for a single streaming feature distribution.

## Update API

```python
result = rtta.FeatureDistributionDriftDetector().update(feature)
```

The `update(...)` call consumes one observation using `feature`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`FeatureDistributionDriftDetector` maintains an adaptive recent window and searches every admissible split for a statistically meaningful difference between the old and new subwindow means. The signal direction is the sign of the best accepted mean shift; accepting a split discards the older prefix.

## Recurrence

Let \(z_t = feature_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
W_t=\operatorname{tail}_{max\_window}(W_{t-1}\cup\{x_t\})
\]

\[
\epsilon(c)=R_t
\sqrt{\frac{1}{2}\log\left(\frac{4}{\delta}\right)
\left(\frac{1}{c}+\frac{1}{|W_t|-c}\right)}
\]

\[
c^\*=\arg\max_c |\bar{x}_{c:|W_t|}-\bar{x}_{1:c}|
\quad \text{s.t.}\quad
|\bar{x}_{c:|W_t|}-\bar{x}_{1:c}|>\epsilon(c)
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(\bar{x}_{c^\*:|W_t|}-\bar{x}_{1:c^\*}), & c^\* \text{ exists}\\
0, & \text{otherwise}
\end{cases}
\]

When a cut is accepted, the older prefix is discarded and the retained suffix
becomes the next adaptive window.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class FeatureDistributionDriftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Concept_drift)
