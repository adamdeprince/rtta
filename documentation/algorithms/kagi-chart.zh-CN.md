# 卡吉图（KagiChart）

## 摘要

`KagiChart` 是一种流式卡吉线：线条沿当前方向跟踪价格，只有当价格向相反方向绝对移动 `reversal` 大小时才会**反转**。输出为当前线位、阳/阴方向以及反转标志。

## 更新 API

```python
import rtta

ind = rtta.KagiChart(reversal=1.0)
result = ind.update(price)
# result.line, result.direction（+1 阳 / -1 阴 / 0 未设定），
# result.reversal（本次逐笔发生方向翻转时为 1，否则为 0）
```

`reversal` 是绝对价格金额（不是百分比），下限为 \(10^{-12}\)。

## 工作原理

卡吉图忽略时间和小幅噪声：只要价格继续沿当前方向移动，线条就会延伸；只有至少达到反转金额的反向移动，才会使阳线（上涨，\(+1\)）转为阴线（下跌，\(-1\)），或反之。经典图表还会根据此前高低点使用粗线和细线；本实现公开线位、有符号方向和反转事件，便于量化使用。

## 递推公式

令 \(\delta=\)`reversal`。首个价格：\(\ell=p_0\)，方向 \(d=0\)，\(\mathrm{reversal}=0\)。

若 \(d\ge0\)（未设方向或为阳）：

\[
\begin{aligned}
p_t > \ell &\Rightarrow \ell \leftarrow p_t
\quad\text{（若 \(d=0\)，同时令 \(d\leftarrow+1\)）},\\
p_t \le \ell - \delta &\Rightarrow d \leftarrow -1,\ \ell \leftarrow p_t,\
\mathrm{reversal}\leftarrow 1.
\end{aligned}
\]

若 \(d<0\)（阴）：

\[
\begin{aligned}
p_t < \ell &\Rightarrow \ell \leftarrow p_t,\\
p_t \ge \ell + \delta &\Rightarrow d \leftarrow +1,\ \ell \leftarrow p_t,\
\mathrm{reversal}\leftarrow 1.
\end{aligned}
\]

输出：\(\mathrm{line}_t=\ell\)，\(\mathrm{direction}_t=d\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class KagiChart` 中实现。结果类型为 `KagiChartResult`。

## 参考资料

- [Investopedia——Kagi Chart](https://www.investopedia.com/terms/k/kagichart.asp)
