# LinearRegressionAngle

## 摘要

`LinearRegressionAngle` 返回滚动线性回归斜率对应的角度。

## 更新 API

```python
result = rtta.LinearRegressionAngle().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动最小二乘所需的充分统计量，加入新样本、移除过期样本，再计算当前回归角度。

## 递推公式

\[
\hat\beta_t=(X_t^\top X_t)^{-1}X_t^\top y_t,\qquad \hat y_t=[1,t]\hat\beta_t
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class LinearRegressionAngle` 中实现。

## 参考资料

- [ChartSchool：Slope](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
