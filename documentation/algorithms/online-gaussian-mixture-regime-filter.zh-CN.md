# OnlineGaussianMixtureRegimeFilter

## 摘要

`OnlineGaussianMixtureRegimeFilter` 是分量数有界的在线高斯混合状态滤波器。

## 更新 API

```python
result = rtta.OnlineGaussianMixtureRegimeFilter().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器根据新观测对各高斯分量的似然更新责任度和权重，再由最匹配的分量表示当前状态。

## 递推公式

\[
r_{t,k}=\frac{w_{t-1,k}p(z_t\mid k)}{\sum_jw_{t-1,j}p(z_t\mid j)}
\]

\[
w_{t,k}=(1-\alpha)w_{t-1,k}+\alpha r_{t,k}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class OnlineGaussianMixtureRegimeFilter` 中实现。

## 参考资料

- [背景资料：混合模型](https://en.wikipedia.org/wiki/Mixture_model)
