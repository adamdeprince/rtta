# ThresholdRegimeDetector

## 摘要

`ThresholdRegimeDetector` 是带上下回滞带的有状态阈值检测器。

## 更新 API

```python
result = rtta.ThresholdRegimeDetector().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

输入值直接作为状态指标。独立的进入与退出阈值使输出在边界附近保持稳定，只有越过相反方向的退出带才会改变状态。

## 递推公式

\[
q_t=value_t
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ThresholdRegimeDetector` 中实现。

## 参考资料

- [背景资料：回滞](https://en.wikipedia.org/wiki/Hysteresis)
