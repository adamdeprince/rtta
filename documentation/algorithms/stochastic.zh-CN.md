# Stochastic

## 摘要

`Stochastic` 计算慢速随机振荡器。

## 更新 API

```python
result = rtta.Stochastic().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标把收盘价在近期高低区间中的位置归一化，再对 %K 进行因果平滑得到慢速 %K 与 %D。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1}),\qquad y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

`update(...)` 返回含 `slowk` 和 `slowd` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Stochastic` 中实现。

## 参考资料

- [ChartSchool：随机振荡器](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full)
