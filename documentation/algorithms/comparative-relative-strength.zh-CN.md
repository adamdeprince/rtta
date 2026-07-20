# 比较相对强弱（ComparativeRelativeStrength）

## 摘要

`ComparativeRelativeStrength` 是 RTTA 对两个序列价格比率的流式实现。它在每个时点返回 \(price_a / price_b\)，不作平滑，也不保存窗口状态。

## 更新 API

```python
value = rtta.ComparativeRelativeStrength().update(price_a, price_b)
```

构造函数没有参数。数组形式的 `batch(...)` 要求 `price_a` 与 `price_b` 数组长度相同。

## 工作原理

两种资产之间的比较强弱（或相对强弱）就是其价格之比。比率上升表示序列 A 的表现优于序列 B；比率下降表示 A 表现逊于 B。交易者常直接绘制该比率，也会对比率应用移动平均线（RTTA 将这种可选的后处理留给调用方）。

它不同于 RSI（“相对强弱指数”）：后者是针对单一序列涨跌幅的振荡器，而本指标是直接的比较比率。

## 递推公式

令 \(a_t\) 和 \(b_t\) 为时刻 \(t\) 的两个价格。

\[
CRS_t = \frac{a_t}{b_t}
\quad\text{（安全除法；若 \(b_t = 0\)，则为 \(0\)）}
\]

指标没有内部状态；每次更新都与此前 K 线无关。

## 实现说明

该计算在 `src/rtta/indicator.cpp` 的 `class ComparativeRelativeStrength` 中实现。

## 参考资料

- [ChartSchool：Price Relative / Relative Strength](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/price-relative-relative-strength)
- [Investopedia：Relative Strength](https://www.investopedia.com/terms/r/relativestrength.asp)
