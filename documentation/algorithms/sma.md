# SMA

## Summary

`SMA` computes a rolling simple moving average over one scalar stream. RTTA
maintains a circular buffer and a rolling sum, so each `update(value)` is
constant-time.

## Update API

```python
value = rtta.SMA(window=30, fillna=False).update(value)
```

`window` is the maximum number of samples in the rolling average.

## Theory Of Operation

A simple moving average treats every sample in the active window equally. RTTA
updates the sum by subtracting the sample that leaves the circular buffer and
adding the new sample.

## Recurrence

Let \(x_t\) be the new sample, \(n\) be `window`, and \(m_t=\min(t+1,n)\). Let
\(x_{t-n}=0\) during the initial warmup before a full expired sample exists.

\[
S_t = S_{t-1} + x_t - x_{t-n}
\]

The returned average is:

\[
SMA_t =
\begin{cases}
S_t / m_t, & \text{during warmup and `fillna=True`} \\
S_t / n, & \text{after the window is full}
\end{cases}
\]

With `fillna=False`, warmup calls return `NaN` while the circular buffer is not
full, but the rolling sum and buffer are still updated.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SMA`.

## Reference

- [ChartSchool: Simple and Exponential Moving Averages](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
