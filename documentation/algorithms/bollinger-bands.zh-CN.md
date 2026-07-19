# BollingerBands

## 摘要

`BollingerBands` 计算以标准差为宽度的移动平均包络。

## 更新 API

```python
result = rtta.BollingerBands().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

布林带在滚动 `SMA` 外包覆滚动 `StdDev` 通道。中轨是局部均值，上轨和下轨分别位于均值上下两个标准差处。

## 递推公式

令 \(z_t = value_t\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
M_t=\operatorname{SMA}_n(x_t), \qquad
S_t=\operatorname{StdDev}_n(x_t)
\]

\[
upper_t=M_t+2S_t, \qquad lower_t=M_t-2S_t
\]

`update(...)` 返回含有 `middle`、`upper` 和 `lower` 字段的结果结构体。

## 组合基础组件

[`SMA`](sma.zh-CN.md)、[`StdDev`](std-dev.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class BollingerBands` 中实现。

## 参考资料

- [ChartSchool：布林带](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands)
