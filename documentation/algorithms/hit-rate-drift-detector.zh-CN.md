# HitRateDriftDetector

## 摘要

`HitRateDriftDetector` 以 EWMA 跟踪未命中率，并用回滞检测命中率退化。

## 更新 API

```python
result = rtta.HitRateDriftDetector().update(hit)
```

`update(...)` 每次接收一个 `hit`；只推进状态时可调用 `advance(...)`。

## 工作原理

非正输入记为未命中，检测器对未命中事件做 EWMA 平滑。数值越高表示命中率越差，再以进入/退出阈值生成稳定状态。

## 递推公式

\[
m_t=\mathbf1[hit_t\le0],\qquad q_t=\alpha m_t+(1-\alpha)q_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class HitRateDriftDetector` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
