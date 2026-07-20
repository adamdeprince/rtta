# 点数图（PointAndFigure）

## 摘要

`PointAndFigure` 是流式点数图（P&F，也称 OX 图）状态：价格按 `box_size` 大小量化到格值，只有当价格向当前方向的反方向移动 `reversal_boxes` 格时，列才会反转。输出为当前格值价格、方向、本次逐笔绘制的格数以及反转标志。

## 更新 API

```python
import rtta

ind = rtta.PointAndFigure(box_size=1.0, reversal_boxes=3)
result = ind.update(price)
# result.box_price, result.direction（+1 X / -1 O / 0 未设定），
# result.boxes（本次逐笔移动的有符号格数）, result.reversal
```

初始格值为 \(\lfloor p_0/b\rfloor\cdot b\)，其中 \(b=\)`box_size`。

## 工作原理

点数图舍弃时间和格内噪声。上涨列（X，\(+1\)）在价格触及上一格时继续向上延伸；下跌列（O，\(-1\)）则向下延伸。只有价格从当前格值反向移动至少 \(R\) 格，列方向才会翻转，格值也会跳到向下取整量化后的目标。本类公开实时格值和每次逐笔的格数增量，而不是完整的列矩阵。

## 递推公式

令 \(b>0\) 为格值大小，\(R=\max(\mathrm{reversal\_boxes},1)\)。当前格值价格为 \(B\)，方向为 \(d\)。

若 \(d\ge0\)（未设方向或上涨）：

\[
\text{当 } p_t \ge B + b \text{ 时重复：}\quad B \leftarrow B + b,\ \Delta \leftarrow \Delta + 1
\quad\text{（若 \(d=0\)，令 \(d\leftarrow+1\)）}.
\]

若没有上涨（\(\Delta=0\)）且 \(p_t\le B-Rb\)：

\[
B^\star = \lfloor p_t / b\rfloor b,\quad
\Delta = -\max\bigl(1, (B - B^\star)/b\bigr),\quad
B \leftarrow B^\star,\ d \leftarrow -1,\ \mathrm{reversal}\leftarrow 1.
\]

若 \(d<0\)（下跌）：

\[
\text{当 } p_t \le B - b \text{ 时重复：}\quad B \leftarrow B - b,\ \Delta \leftarrow \Delta - 1.
\]

若没有下跌且 \(p_t\ge B+Rb\)：

\[
B^\star = \lfloor p_t / b\rfloor b,\quad
\Delta = \max\bigl(1, (B^\star - B)/b\bigr),\quad
B \leftarrow B^\star,\ d \leftarrow +1,\ \mathrm{reversal}\leftarrow 1.
\]

输出：\(\mathrm{box\_price}=B\)、\(\mathrm{boxes}=\Delta\)、\(\mathrm{direction}=d\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class PointAndFigure` 中实现。结果类型为 `PointAndFigureResult`。

## 参考资料

- [Investopedia——Point-and-Figure Charting](https://www.investopedia.com/terms/p/pointandfigurechart.asp)
