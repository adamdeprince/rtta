# LinearRegression

## 摘要

`LinearRegression` 返回滚动最小二乘回归的当前拟合值。

## 更新 API

```python
result = rtta.LinearRegression().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动充分统计量，每次加入最新样本、移除过期样本，并重新计算当前回归结果。

## 递推公式

\[
\hat\beta_t=(X_t^\top X_t)^{-1}X_t^\top y_t,\qquad \hat y_t=[1,t]\hat\beta_t
\]

`update(...)` 返回含 `value`、`slope`、`intercept`、`angle` 和 `tsf` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class LinearRegression` 中实现。

## 参考资料

- [ChartSchool：Slope](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
