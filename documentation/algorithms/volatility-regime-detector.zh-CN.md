# VolatilityRegimeDetector

## 摘要

`VolatilityRegimeDetector` 用收盘价变化的 EWMA 波动率和高低回滞带检测波动状态。

## 更新 API

```python
result = rtta.VolatilityRegimeDetector().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器对相邻收盘价变化的平方进行 EWMA 平滑，取平方根得到波动率，再应用双向进入/退出回滞。

## 递推公式

\[
\Delta_t=close_t-close_{t-1},\qquad v_t=(1-\alpha)(v_{t-1}+\alpha\Delta_t^2),\qquad q_t=\sqrt{v_t}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolatilityRegimeDetector` 中实现。

## 参考资料

- [ChartSchool：标准差与波动率](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
