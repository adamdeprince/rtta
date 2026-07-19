# EDDM

## 摘要

`EDDM` 根据预测误差之间的间距进行早期漂移检测。

## 更新 API

```python
result = rtta.EDDM().update(error)
```

`update(...)` 每次接收一个 `error`；只推进状态时可调用 `advance(...)`。

## 工作原理

正输入视为分类误差。检测器跟踪相邻误差之间的样本距离，并把当前距离统计量相对历史最佳值的退化程度用于区分警告和漂移。

## 递推公式

\[
d_k=i_k-i_{k-1}\quad\text{其中 }i_k\text{ 为满足 }error_{i_k}>0\text{ 的样本下标}
\]

\[
\bar d_k=\bar d_{k-1}+\frac{d_k-\bar d_{k-1}}k,\qquad s^2_{d,k}=\frac1{k-1}\sum_{j=1}^k(d_j-\bar d_k)^2
\]

\[
M_k=\bar d_k+2s_{d,k},\qquad \rho_k=\frac{M_k}{\max_{j\le k}M_j}
\]

\[
y_t=\begin{cases}1,&\rho_k<\rho_{drift}\\0.5,&\rho_k<\rho_{warning}\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class EDDM` 中实现。

## 参考资料

- [背景资料：概念漂移](https://en.wikipedia.org/wiki/Concept_drift)
