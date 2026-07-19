# RealizedVarianceRegimeDetector

## 摘要

`RealizedVarianceRegimeDetector` 根据相邻收盘价变化的平方和检测滚动已实现方差状态。

## 更新 API

```python
result = rtta.RealizedVarianceRegimeDetector().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器计算窗口内收盘价变化平方的均值，再用双向进入/退出回滞生成稳定的方差状态。

## 递推公式

\[
\Delta_t=close_t-close_{t-1},\qquad q_t=\frac1n\sum_{i\in W_t}\Delta_i^2
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RealizedVarianceRegimeDetector` 中实现。

## 参考资料

- [背景资料：已实现方差](https://en.wikipedia.org/wiki/Realized_variance)
