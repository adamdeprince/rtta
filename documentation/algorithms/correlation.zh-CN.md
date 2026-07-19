# Correlation

## 摘要

`Correlation` 计算两个序列之间的滚动 Pearson 相关系数。

## 更新 API

```python
result = rtta.Correlation().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

该类维护滚动充分统计量。每次更新加入最新样本、移除过期样本，再从保存的累计值计算当前相关系数。

## 递推公式

令 \(z_t=(real0_t,real1_t)\) 为一次更新接收的观测。

\[
\mu^x_t,\mu^y_t,\sigma^2_{x,t},\sigma^2_{y,t},c_{xy,t}=\operatorname{rollstats}(x_t,y_t,n)
\]

\[
\rho_t=\frac{c_{xy,t}}{\sigma_{x,t}\sigma_{y,t}},\qquad \beta_t=\frac{c_{xy,t}}{\sigma^2_{x,t}}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Correlation` 中实现。

## 参考资料

- [ChartSchool：相关系数](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient)
