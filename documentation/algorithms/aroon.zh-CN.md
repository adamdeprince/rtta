# Aroon

## 摘要

`Aroon` 根据近期最高点和最低点出现的时间，计算 Aroon Up 与 Aroon Down 趋势年龄指标。

## 更新 API

```python
result = rtta.Aroon().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

`Aroon` 维护滚动极值。每接收一个样本，C++ 状态只更新一次相应的窗口统计量。

## 递推公式

令 \(z_t = (high_t, low_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

`update(...)` 返回含有 `down` 和 `up` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Aroon` 中实现。

## 参考资料

- [ChartSchool：Aroon](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon)
