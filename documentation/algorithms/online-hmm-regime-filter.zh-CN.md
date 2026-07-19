# OnlineHMMRegimeFilter

## 摘要

`OnlineHMMRegimeFilter` 是转移持续性固定的在线高斯隐马尔可夫状态滤波器。

## 更新 API

```python
result = rtta.OnlineHMMRegimeFilter().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器先用转移矩阵预测状态概率，再结合新观测在各状态下的似然进行归一化更新。固定的持续性使状态不会因轻微反转立即切换。

## 递推公式

\[
\tilde\pi_t=A^\top\pi_{t-1}
\]

\[
\pi_t(i)=\frac{\tilde\pi_t(i)p(z_t\mid i)}{\sum_j\tilde\pi_t(j)p(z_t\mid j)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class OnlineHMMRegimeFilter` 中实现。

## 参考资料

- [背景资料：隐马尔可夫模型](https://en.wikipedia.org/wiki/Hidden_Markov_model)
