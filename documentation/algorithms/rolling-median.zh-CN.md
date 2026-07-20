# 滚动中位数（RollingMedian）

## 摘要

`RollingMedian` 是 RTTA 对固定长度滚动窗口中位数的流式实现。窗口大小为奇数时，返回居中的次序统计量；为偶数时，返回两个中央次序统计量的平均值。

## 更新 API

```python
value = rtta.RollingMedian(window=14, fillna=True).update(value)
```

当 `fillna=False` 时，在缓冲区填满之前输出为 `NaN`。当 `fillna=True` 时，预热期间使用目前已取得的样本计算中位数。

## 工作原理

中位数是一种稳健的位置估计量：单个尖峰对中位数的影响远小于对均值的影响。流式中位数需要计算当前窗口的次序统计量。RTTA 把窗口复制到临时向量，再使用 `std::nth_element` 找到中央元素，而无需完整排序。

## 递推公式

令 \(x_t\) 为输入，\(n\) 为构造函数的 `window`。维护一个容量为 \(n\) 的 FIFO 缓冲区。每次推入后，令 \(m\) 为当前缓冲区大小，并令 \(\{y_1\le\cdots\le y_m\}\) 在概念上表示排好序的缓冲区内容。

\[
\operatorname{Median}_t =
\begin{cases}
y_{(m+1)/2}, & m \text{ 为奇数} \\[4pt]
\tfrac12\bigl(y_{m/2} + y_{m/2+1}\bigr), & m \text{ 为偶数}
\end{cases}
\]

（次序统计量使用从 1 开始的索引。）实现细节：当 \(m\) 为偶数时，RTTA 先用 `nth_element` 在索引 \(m/2\) 处找到较大的中央值，再在索引 \(m/2-1\) 处找到较小的中央值，最后取平均——与上述公式等价。

## 实现说明

该计算在 `src/rtta/indicator.cpp` 的 `class RollingMedian` 中实现。每次更新都会把临时存储区调整为当前缓冲区大小。

## 参考资料

- [Wikipedia：Median](https://en.wikipedia.org/wiki/Median)
- [Investopedia：Median](https://www.investopedia.com/terms/m/median.asp)
