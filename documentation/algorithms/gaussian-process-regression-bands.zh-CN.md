# GaussianProcessRegressionBands

## 摘要

`GaussianProcessRegressionBands` 计算滚动 RBF 核高斯过程的后验均值和不确定性带。

## 更新 API

```python
result = rtta.GaussianProcessRegressionBands(window=16).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动窗口所需统计量，加入最新样本并移除过期样本，再计算当前后验均值和方差。

## 递推公式

\[
K_{ij}=k(x_i,x_j)+\sigma^2\delta_{ij}
\]

\[
\mu_t=k_t^\top K^{-1}y,\qquad \sigma_t^2=k(z_t,z_t)-k_t^\top K^{-1}k_t
\]

`update(...)` 返回含 `middle`、`upper` 和 `lower` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class GaussianProcessRegressionBands` 中实现。

## 参考资料

- [Gaussian Process Regression](https://www.luxalgo.com/library/indicator/machine-learning-gaussian-process-regression/)
