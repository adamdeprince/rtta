# RollingMedian

## Summary

`RollingMedian` is RTTA's streaming median of a fixed-length rolling window. For
odd-sized windows it returns the middle order statistic; for even-sized windows
it returns the average of the two central order statistics.

## Update API

```python
value = rtta.RollingMedian(window=14, fillna=True).update(value)
```

With `fillna=False`, output is `NaN` until the buffer is full. During warmup
with `fillna=True`, the median is computed over the samples seen so far.

## Theory Of Operation

The median is a robust location estimator: a single spike does not move it as
much as it moves a mean. Streaming medians require order statistics of the
current window. RTTA copies the window into a scratch vector and uses
`std::nth_element` to find the central element(s) without a full sort.

## Recurrence

Let \(x_t\) be the input and \(n\) the constructor `window`. Maintain a FIFO
buffer of capacity \(n\). After each push, let \(m\) be the current buffer size
and let \(\{y_1 \le \cdots \le y_m\}\) be the sorted buffer contents (conceptually).

\[
\operatorname{Median}_t =
\begin{cases}
y_{(m+1)/2}, & m \text{ odd} \\[4pt]
\tfrac12\bigl(y_{m/2} + y_{m/2+1}\bigr), & m \text{ even}
\end{cases}
\]

(1-based order statistics). Implementation detail: for even \(m\), RTTA finds
the upper middle via `nth_element` at index \(m/2\), then finds the lower middle
at index \(m/2 - 1\), and averages them — equivalent to the formula above.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class RollingMedian`. Scratch storage is resized to the current buffer size
each update.

## Reference

- [Wikipedia: Median](https://en.wikipedia.org/wiki/Median)
- [Investopedia: Median](https://www.investopedia.com/terms/m/median.asp)
