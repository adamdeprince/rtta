# CorrelationRegimeDetector

## 摘要

`CorrelationRegimeDetector` 是带上下回滞带的有状态滚动相关性检测器。

## 更新 API

```python
result = rtta.CorrelationRegimeDetector().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器计算两个序列的滚动相关系数，再用进入/退出回滞生成稳定的正相关、中性或负相关状态。

## 递推公式

令 \(z_t=(real0_t,real1_t)\) 为一次更新接收的观测。

\[
q_t=\rho_t=\frac{n\sum xy-\sum x\sum y}{\sqrt{(n\sum x^2-(\sum x)^2)(n\sum y^2-(\sum y)^2)}}
\]

各项累计值在指定滚动窗口内维护。

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CorrelationRegimeDetector` 中实现。

## 参考资料

- [ChartSchool：相关系数](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient)
