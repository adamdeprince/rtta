# BetaRegimeDetector

## 摘要

`BetaRegimeDetector` 是带有上下回滞带的有状态滚动 beta 状态检测器。

## 更新 API

```python
result = rtta.BetaRegimeDetector().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器先从当前观测和滚动状态计算 beta，再应用进入/退出回滞。只有 beta 越过相反方向的退出阈值，输出状态才会改变。

## 递推公式

令 \(z_t = (real0_t, real1_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
q_t=\beta_t=
\frac{n\sum xy-\sum x\sum y}{n\sum y^2-(\sum y)^2}
\]

各项累计值在指定滚动窗口内维护；C++ 中的 beta 等于 `real0` 与 `real1` 的协方差除以 `real1` 的方差。

\[
r_t =
\begin{cases}
1, & r_{t-1} \le 0 \text{ 且 } q_t \ge u_e \\
0, & r_{t-1} = 1 \text{ 且 } q_t \le u_x \\
-1, & r_{t-1} \ge 0 \text{ 且 } q_t \le \ell_e \\
0, & r_{t-1} = -1 \text{ 且 } q_t \ge \ell_x \\
r_{t-1}, & \text{否则}
\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class BetaRegimeDetector` 中实现。

## 参考资料

- [背景资料：Beta](https://www.investopedia.com/terms/b/beta.asp)
