# StickyHMMRegimeFilter

## 摘要

`StickyHMMRegimeFilter` 是自转移概率较高的在线高斯 HMM 状态滤波器。

## 更新 API

```python
result = rtta.StickyHMMRegimeFilter().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器先由转移矩阵预测状态概率，再结合新观测似然更新。较高的自转移概率使状态保持黏性，轻微反转不会立即翻转输出。

## 递推公式

\[
\tilde\pi_t=A^\top\pi_{t-1},\qquad \pi_t(i)=\frac{\tilde\pi_t(i)p(z_t\mid i)}{\sum_j\tilde\pi_t(j)p(z_t\mid j)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class StickyHMMRegimeFilter` 中实现。

## 参考资料

- [背景资料：隐马尔可夫模型](https://en.wikipedia.org/wiki/Hidden_Markov_model)
