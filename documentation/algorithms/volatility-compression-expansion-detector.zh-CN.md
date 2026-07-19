# VolatilityCompressionExpansionDetector

## 摘要

`VolatilityCompressionExpansionDetector` 比较短期与长期 EWMA 波动率，以识别压缩和扩张状态。

## 更新 API

```python
result = rtta.VolatilityCompressionExpansionDetector().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器分别维护短期与长期收益方差，以两者波动率之比作为状态指标，再应用双向回滞。

## 递推公式

\[
r_t=\frac{close_t-close_{t-1}}{close_{t-1}}
\]

\[
v^S_t=(1-\alpha_S)(v^S_{t-1}+\alpha_Sr_t^2),\qquad v^L_t=(1-\alpha_L)(v^L_{t-1}+\alpha_Lr_t^2)
\]

\[
q_t=\frac{\sqrt{\max(v^S_t,\epsilon)}}{\sqrt{\max(v^L_t,\epsilon)}}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolatilityCompressionExpansionDetector` 中实现。

## 参考资料

- [ChartSchool：标准差与波动率](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
