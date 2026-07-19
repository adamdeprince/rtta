# BoundedBOCPD

## 摘要

`BoundedBOCPD` 是采用恒定风险率、内存有界的贝叶斯在线变点检测器。

## 更新 API

```python
result = rtta.BoundedBOCPD().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器为每条观测更新流式后验分数，并依据阈值确定当前状态。状态有意保持一定黏性，轻微反转不会立即翻转输出。

## 递推公式

令 \(z_t = value_t\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
R_t(r+1) = R_{t-1}(r)(1-h)p(z_t\mid r)
\]

\[
R_t(0)=\sum_r R_{t-1}(r)h\,p(z_t\mid r)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class BoundedBOCPD` 中实现。

## 参考资料

- [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)
