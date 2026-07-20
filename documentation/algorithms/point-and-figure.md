# PointAndFigure

## Summary

`PointAndFigure` is a streaming point-and-figure (P&F) chart state: price is quantized to
a box grid of size `box_size`, and a column reverses only after a move of
`reversal_boxes` boxes against the current direction. Outputs are the current box price,
direction, boxes drawn this tick, and a reversal flag.

## Update API

```python
import rtta

ind = rtta.PointAndFigure(box_size=1.0, reversal_boxes=3)
result = ind.update(price)
# result.box_price, result.direction (+1 X / -1 O / 0 unset),
# result.boxes (signed count moved this tick), result.reversal
```

Initial box is \(\lfloor p_0 / b\rfloor \cdot b\) with \(b=\)`box_size`.

## Theory Of Operation

P&F charts discard time and intra-box noise. An up column (X, \(+1\)) extends whenever
price reaches the next higher box; a down column (O, \(-1\)) extends to lower boxes. A
reversal requires a move of at least \(R\) boxes from the current box price, after which
the column direction flips and the box jumps to the floor-quantized target. This class
exposes the live box level and per-tick box increments rather than a full column matrix.

## Recurrence

Let \(b > 0\) be box size and \(R = \max(\mathrm{reversal\_boxes}, 1)\). Current box
price \(B\), direction \(d\).

If \(d \ge 0\) (flat or rising):

\[
\text{while } p_t \ge B + b:\quad B \leftarrow B + b,\ \Delta \leftarrow \Delta + 1
\quad\text{(if \(d=0\), set \(d\leftarrow +1\))}.
\]

If no ascent (\(\Delta=0\)) and \(p_t \le B - R b\):

\[
B^\star = \lfloor p_t / b\rfloor b,\quad
\Delta = -\max\bigl(1, (B - B^\star)/b\bigr),\quad
B \leftarrow B^\star,\ d \leftarrow -1,\ \mathrm{reversal}\leftarrow 1.
\]

If \(d < 0\) (falling):

\[
\text{while } p_t \le B - b:\quad B \leftarrow B - b,\ \Delta \leftarrow \Delta - 1.
\]

If no descent and \(p_t \ge B + R b\):

\[
B^\star = \lfloor p_t / b\rfloor b,\quad
\Delta = \max\bigl(1, (B^\star - B)/b\bigr),\quad
B \leftarrow B^\star,\ d \leftarrow +1,\ \mathrm{reversal}\leftarrow 1.
\]

Outputs: \(\mathrm{box\_price}=B\), \(\mathrm{boxes}=\Delta\), \(\mathrm{direction}=d\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PointAndFigure`.
Result type is `PointAndFigureResult`.

## Reference

- [Investopedia — Point-and-Figure Charting](https://www.investopedia.com/terms/p/pointandfigurechart.asp)
