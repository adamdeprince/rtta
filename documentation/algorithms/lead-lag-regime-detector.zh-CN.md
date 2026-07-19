# LeadLagRegimeDetector

## 摘要

`LeadLagRegimeDetector` 用 EWMA 交叉滞后统计量判断两个序列中哪一个领先。

## 更新 API

```python
result = rtta.LeadLagRegimeDetector().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器比较 \(x\) 的滞后变动对当前 \(y\) 的解释力与反方向关系，再对差值做 EWMA 归一化并应用双向回滞。

## 递推公式

\[
\Delta x_t=x_t-x_{t-1},\qquad \Delta y_t=y_t-y_{t-1}
\]

\[
a_t=\Delta x_{t-1}\Delta y_t,\qquad b_t=\Delta y_{t-1}\Delta x_t
\]

\[
S_t=\alpha(a_t-b_t)+(1-\alpha)S_{t-1},\qquad C_t=\alpha(|a_t|+|b_t|)+(1-\alpha)C_{t-1},\qquad q_t=\frac{S_t}{\max(C_t,\epsilon)}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class LeadLagRegimeDetector` 中实现。

## 参考资料

- [背景资料：互相关](https://en.wikipedia.org/wiki/Cross-correlation)
