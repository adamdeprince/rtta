# HiddenSemiMarkovRegimeFilter

## 摘要

`HiddenSemiMarkovRegimeFilter` 是带有有界持续期偏置的在线高斯隐半马尔可夫式状态滤波器。

## 更新 API

```python
result = rtta.HiddenSemiMarkovRegimeFilter().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把每条观测转换为流式似然分数并更新状态概率。持续期偏置使状态具有一定黏性，轻微反转不会立刻翻转输出。

## 递推公式

\[
\tilde{\pi}_t=A^\top\pi_{t-1}
\]

\[
\pi_t(i)=\frac{\tilde{\pi}_t(i)p(z_t\mid i)}{\sum_j\tilde{\pi}_t(j)p(z_t\mid j)}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class HiddenSemiMarkovRegimeFilter` 中实现。

## 参考资料

- [背景资料：隐半马尔可夫模型](https://en.wikipedia.org/wiki/Hidden_semi-Markov_model)
