# 效率比率（EfficiencyRatio）

## 摘要

`EfficiencyRatio` 是 RTTA 对考夫曼效率比率的流式实现：滚动窗口内净方向移动与路径长度之比。

## 更新 API

```python
result = rtta.EfficiencyRatio().update(close)
```

每次调用 `update(...)` 都使用 `close` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`EfficiencyRatio` 是驱动 [`Kama`](kama.zh-CN.md) 的方向/噪声比率。数值接近 1 表示走势方向清晰；接近 0 表示路径嘈杂而净位移很小。

## 递推公式

令 \(c_t=close_t\)，\(n\) 为窗口长度。

\[
ER_t = \frac{|c_t - c_{t-n}|}{\sum_{i=0}^{n-1} |c_{t-i} - c_{t-i-1}|}
\]

分母是一步绝对变化的滚动和；分子使用 \(n\) 个样本前的收盘价，与 KAMA 内部的效率项一致。

返回值为当前的标量指标值。

## 组合使用的基础指标

[`Kama`](kama.zh-CN.md)

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class EfficiencyRatio` 中实现。

## 参考资料

- [背景资料](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama)
