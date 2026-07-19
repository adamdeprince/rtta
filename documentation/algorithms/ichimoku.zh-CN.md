# Ichimoku

## 摘要

`Ichimoku` 计算一目均衡表的转换线、基准线和先行带分量。

## 更新 API

```python
result = rtta.Ichimoku().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护多个周期的滚动高低区间，并以各区间中点构造转换线、基准线和云层分量。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

`update(...)` 返回含 `conversion`、`base`、`span_a` 和 `span_b` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Ichimoku` 中实现。

## 参考资料

- [ChartSchool：一目均衡表](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud)
