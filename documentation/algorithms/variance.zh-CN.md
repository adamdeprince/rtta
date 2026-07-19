# Variance

## 摘要

`Variance` 计算滚动方差。

## 更新 API

```python
result = rtta.Variance().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护窗口所需的充分统计量，每次加入新样本、移除过期样本，并重新计算当前方差。

## 递推公式

\[
\mu_t=\frac1{|W_t|}\sum_{i\in W_t}x_i
\]

\[
\sigma_t^2=\frac1{|W_t|}\sum_{i\in W_t}(x_i-\mu_t)^2
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Variance` 中实现。

## 参考资料

- [ChartSchool：标准差与波动率](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
