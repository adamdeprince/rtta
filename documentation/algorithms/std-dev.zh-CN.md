# StdDev

## 摘要

`StdDev` 计算滚动标准差。

## 更新 API

```python
result = rtta.StdDev(window=5).update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次更新把新观测加入滚动窗口、移除过期值，并从窗口均值计算总体方差及标准差。

## 递推公式

\[
\mu_t=\frac1{|W_t|}\sum_{i\in W_t}x_i
\]

\[
\sigma_t^2=\frac1{|W_t|}\sum_{i\in W_t}(x_i-\mu_t)^2
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class StdDev` 中实现。

## 参考资料

- [ChartSchool：标准差与波动率](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
