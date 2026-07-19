# ResidualDriftDetector

## 摘要

`ResidualDriftDetector` 用 EWMA 残差 z 分数检测漂移，并通过双向回滞输出符号。

## 更新 API

```python
result = rtta.ResidualDriftDetector().update(residual)
```

`update(...)` 每次接收一个 `residual`；只推进状态时可调用 `advance(...)`。

## 工作原理

当前残差先用此前样本估计的 EWMA 均值和方差标准化，再通过双向进入/退出回滞生成持续的漂移方向。

## 递推公式

\[
q_t=\frac{residual_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(residual_t-\mu_{t-1}),\qquad \sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(residual_t-\mu_{t-1})^2)
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ResidualDriftDetector` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
