# VerticalHorizontalFilter

## Summary

`VerticalHorizontalFilter` is RTTA's streaming Vertical Horizontal Filter (VHF):
the net close range over a window divided by the sum of absolute close-to-close
moves in that window. High VHF indicates trending price; low VHF indicates
choppy sideways action.

## Update API

```python
value = rtta.VerticalHorizontalFilter(window=28, fillna=True).update(close)
```

With `fillna=False`, output is `NaN` until the window is full. With fewer than
two samples and `fillna=True`, the value is `0.0`.

## Theory Of Operation

Adam White's VHF compares the straight-line distance price has traveled
(highest close minus lowest close) to the total path length of successive
closes. In a clean trend the path is mostly one-directional, so the ratio is
large; in a range the path zigzags and the ratio shrinks. VHF is often used as
a regime filter: prefer trend-following tools when VHF is high, oscillators when
low.

## Recurrence

Let \(c_t\) be close and \(n\) be `window` (default \(28\), minimum \(2\)).
Maintain a FIFO buffer of the last \(\min(t+1, n)\) closes. Let \(m\) be the
buffer size. If \(m < 2\), return \(0\) (fillna) or `NaN`.

Index the buffer as \(c^{(0)},\ldots,c^{(m-1)}\) oldest to newest:

\[
H_t = \max_{0\le i < m} c^{(i)}, \qquad
L_t = \min_{0\le i < m} c^{(i)}
\]

\[
Path_t = \sum_{i=1}^{m-1} \bigl|c^{(i)} - c^{(i-1)}\bigr|
\]

\[
VHF_t = \frac{H_t - L_t}{Path_t}
\quad\text{(safe divide)}
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class VerticalHorizontalFilter`. High, low, and path are recomputed by a full
pass over the buffer each update.

## Reference

- [Investopedia: Vertical Horizontal Filter](https://www.investopedia.com/terms/v/vhf.asp)
- [TradingPedia: Vertical Horizontal Filter](https://www.tradingpedia.com/forex-trading-indicators/vertical-horizontal-filter/)
