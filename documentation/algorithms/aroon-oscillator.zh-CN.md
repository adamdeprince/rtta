# AroonOscillator

## 摘要

`AroonOscillator` 计算 Aroon Up 与 Aroon Down 之差。

## 更新 API

```python
result = rtta.AroonOscillator().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

该指标把近期方向性变化归一化为振荡值。实现只保留因果的滚动或平滑状态，并据此计算当前值。

## 递推公式

令 \(z_t = (high_t, low_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}
{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AroonOscillator` 中实现。

## 参考资料

- [ChartSchool：Aroon 振荡器](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon-oscillator)
