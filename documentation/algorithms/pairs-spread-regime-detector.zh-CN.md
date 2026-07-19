# PairsSpreadRegimeDetector

## 摘要

`PairsSpreadRegimeDetector` 以流式 EWMA 对冲比率和残差 z 分数检测配对价差状态。

## 更新 API

```python
result = rtta.PairsSpreadRegimeDetector().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器在线估计对冲比率与截距，对当前配对残差做 z 分数标准化，再通过双向回滞输出价差状态。

## 递推公式

\[
\beta_t=\frac{C^{xy}_t}{V^y_t},\qquad \alpha_t=\mu^x_t-\beta_t\mu^y_t
\]

\[
e_t=x_t-(\beta_ty_t+\alpha_t),\qquad q_t=\frac{e_t-\bar e_{t-1}}{\sqrt{\max(s^2_{e,t-1},\epsilon)}}
\]

\[
\bar e_t=\bar e_{t-1}+\eta(e_t-\bar e_{t-1}),\qquad s^2_{e,t}=(1-\eta)(s^2_{e,t-1}+\eta(e_t-\bar e_{t-1})^2)
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class PairsSpreadRegimeDetector` 中实现。

## 参考资料

- [背景资料：统计套利](https://en.wikipedia.org/wiki/Statistical_arbitrage)
