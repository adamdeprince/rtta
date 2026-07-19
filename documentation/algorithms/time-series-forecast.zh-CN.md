# TimeSeriesForecast

## 摘要

`TimeSeriesForecast` 以滚动线性回归进行时间序列预测。

## 更新 API

```python
result = rtta.TimeSeriesForecast().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动最小二乘所需的充分统计量，加入新样本、移除过期样本，再外推当前拟合值。

## 递推公式

\[
\hat\beta_t=(X_t^\top X_t)^{-1}X_t^\top y_t,\qquad \hat y_t=[1,t]\hat\beta_t
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TimeSeriesForecast` 中实现。

## 参考资料

- [ChartSchool：Slope](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
