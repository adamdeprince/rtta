# 垂直水平过滤器（VerticalHorizontalFilter）

## 摘要

`VerticalHorizontalFilter` 是 RTTA 对垂直水平过滤器（VHF）的流式实现：窗口内收盘价净区间除以该窗口内收盘到收盘绝对变动之和。VHF 较高表示价格呈趋势，较低则表示横盘反复震荡。

## 更新 API

```python
value = rtta.VerticalHorizontalFilter(window=28, fillna=True).update(close)
```

当 `fillna=False` 时，在窗口填满之前输出为 `NaN`。样本少于两个且 `fillna=True` 时，数值为 `0.0`。

## 工作原理

Adam White 的 VHF 比较价格移动的直线距离（最高收盘价减最低收盘价）与连续收盘价的总路径长度。趋势清晰时，路径大多沿同一方向，因而比率较大；在区间行情中，路径来回折返，比率随之缩小。VHF 常用作市场状态过滤器：数值较高时偏好趋势跟随工具，较低时偏好振荡器。

## 递推公式

令 \(c_t\) 为收盘价，\(n\) 为 `window`（默认 \(28\)，最小为 \(2\)）。维护最近 \(\min(t+1,n)\) 个收盘价的 FIFO 缓冲区。令 \(m\) 为缓冲区大小。若 \(m<2\)，返回 \(0\)（填充值）或 `NaN`。

按从旧到新把缓冲区记为 \(c^{(0)},\ldots,c^{(m-1)}\)：

\[
H_t = \max_{0\le i < m} c^{(i)}, \qquad
L_t = \min_{0\le i < m} c^{(i)}
\]

\[
Path_t = \sum_{i=1}^{m-1} \bigl|c^{(i)} - c^{(i-1)}\bigr|
\]

\[
VHF_t = \frac{H_t - L_t}{Path_t}
\quad\text{（安全除法）}
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class VerticalHorizontalFilter` 中实现。每次更新都会完整遍历缓冲区，重新计算最高值、最低值和路径。

## 参考资料

- [Investopedia：Vertical Horizontal Filter](https://www.investopedia.com/terms/v/vhf.asp)
- [TradingPedia：Vertical Horizontal Filter](https://www.tradingpedia.com/forex-trading-indicators/vertical-horizontal-filter/)
