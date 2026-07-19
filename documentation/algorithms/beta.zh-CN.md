# Beta

## 摘要

`Beta` 计算一个序列相对于另一个序列的滚动 beta。

## 更新 API

```python
result = rtta.Beta().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

`Beta` 维护计算滚动统计量所需的充分统计量。每次更新加入最新样本、移除过期样本，并用保存的累计值重新计算结果。

## 递推公式

令 \(z_t = (real0_t, real1_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
\mu^x_t,\mu^y_t,\sigma^2_{x,t},\sigma^2_{y,t},c_{xy,t}
= \operatorname{rollstats}(x_t,y_t,n)
\]

\[
\rho_t=\frac{c_{xy,t}}{\sigma_{x,t}\sigma_{y,t}}, \qquad
\beta_t=\frac{c_{xy,t}}{\sigma^2_{x,t}}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Beta` 中实现。

## 参考资料

- [背景资料：Beta](https://www.investopedia.com/terms/b/beta.asp)
