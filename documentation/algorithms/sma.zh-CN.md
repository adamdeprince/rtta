# SMA

## 摘要

`SMA` 计算单一标量数据流的滚动简单移动平均。RTTA 维护一个环形缓冲区和滚动总和，因此每次 `update(value)` 的时间复杂度均为常数。

## 更新 API

```python
value = rtta.SMA(window=30, fillna=False).update(value)
```

`window` 指定滚动平均最多包含多少个样本。

## 工作原理

简单移动平均对当前窗口内的每个样本赋予相同权重。RTTA 更新总和时，先减去即将离开环形缓冲区的样本，再加上新样本。

## 递推公式

令 \(x_t\) 为新样本，\(n\) 为 `window`，并令 \(m_t=\min(t+1,n)\)。在初始预热阶段，尚无完整的过期样本时，取 \(x_{t-n}=0\)。

\[
S_t = S_{t-1} + x_t - x_{t-n}
\]

返回的平均值为：

\[
SMA_t =
\begin{cases}
S_t / m_t, & \text{预热阶段且 `fillna=True`} \\
S_t / n, & \text{窗口填满后}
\end{cases}
\]

当 `fillna=False` 时，只要环形缓冲区尚未填满，预热调用就返回 `NaN`；滚动总和和缓冲区本身仍会正常更新。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SMA` 中实现。

## 参考资料

- [ChartSchool：简单移动平均与指数移动平均](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
