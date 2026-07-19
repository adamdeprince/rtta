# MinusDirectionalMovement

## 摘要

`MinusDirectionalMovement` 计算负方向运动。

## 更新 API

```python
result = rtta.MinusDirectionalMovement().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

该指标属于 Wilder 方向运动体系。它比较当前 K 线相对上一根 K 线的上下延伸，并对负方向运动进行 Wilder 平滑。

## 递推公式

\[
up_t=high_t-high_{t-1},\qquad down_t=low_{t-1}-low_t
\]

\[
DM^+_t=\begin{cases}up_t,&up_t>down_t\text{ 且 }up_t>0\\0,&\text{否则}\end{cases}
\]

\[
DM^-_t=\begin{cases}down_t,&down_t>up_t\text{ 且 }down_t>0\\0,&\text{否则}\end{cases}
\]

\[
y_t=\operatorname{WilderEMA}_n(DM^{\pm}_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MinusDirectionalMovement` 中实现。

## 参考资料

- [ChartSchool：平均趋向指数 ADX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
