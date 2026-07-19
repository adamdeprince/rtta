# ParticleFilterTrend

## 摘要

`ParticleFilterTrend` 是采用确定性随机种子和 Laplace 观测似然的粒子趋势滤波器，并输出有效样本量。

## 更新 API

```python
result = rtta.ParticleFilterTrend(particles=64).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次更新先按状态模型推进全部粒子，再根据新收盘价的 Laplace 似然重设权重并归一化；有效样本量用于诊断权重退化。

## 递推公式

\[
x_t^{(i)}=f(x_{t-1}^{(i)})+\epsilon_t^{(i)}
\]

\[
w_t^{(i)}\propto w_{t-1}^{(i)}p(z_t\mid x_t^{(i)}),\qquad \sum_iw_t^{(i)}=1
\]

`update(...)` 返回含 `trend`、`velocity`、`signal` 和 `effective_sample_size` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ParticleFilterTrend` 中实现。

## 参考资料

- [Trend-Following Filters, Part 4](https://alphaarchitect.com/trend-following-filters-part-4/)
