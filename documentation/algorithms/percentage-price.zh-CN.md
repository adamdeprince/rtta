# PercentagePrice

## 摘要

`PercentagePrice` 计算百分比价格振荡器 PPO。

## 更新 API

```python
result = rtta.PercentagePrice().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以慢速平均为基准，把快慢移动平均之差归一化为百分比，并计算信号线与柱状差值。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

`update(...)` 返回含 `ppo`、`signal` 和 `histogram` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class PercentagePrice` 中实现。

## 参考资料

- [ChartSchool：PPO](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-price-oscillator-ppo)
