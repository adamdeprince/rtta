# OnlineMarkovSwitchingVolatilityFilter

## 摘要

`OnlineMarkovSwitchingVolatilityFilter` 对相邻收盘价变化应用在线双状态马尔可夫切换波动率模型。

## 更新 API

```python
result = rtta.OnlineMarkovSwitchingVolatilityFilter().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器维护两个潜在波动状态的在线概率，以此前概率和转移矩阵进行预测，再用新收益观测的似然更新并归一化。

## 递推公式

\[
\tilde\pi_t=A^\top\pi_{t-1}
\]

\[
\pi_t(i)=\frac{\tilde\pi_t(i)p(z_t\mid i)}{\sum_j\tilde\pi_t(j)p(z_t\mid j)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class OnlineMarkovSwitchingVolatilityFilter` 中实现。

## 参考资料

- [背景资料：马尔可夫切换模型](https://en.wikipedia.org/wiki/Markov-switching_model)
