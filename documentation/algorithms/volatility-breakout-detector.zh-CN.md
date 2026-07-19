# VolatilityBreakoutDetector

## 摘要

`VolatilityBreakoutDetector` 用 EWMA z 分数检测异常大的相邻收盘价波动突破。

## 更新 API

```python
result = rtta.VolatilityBreakoutDetector().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

绝对单周期收益先用此前样本估计的 EWMA 均值和方差标准化，再以上侧回滞生成突破状态。

## 递推公式

\[
m_t=\left|\frac{close_t-close_{t-1}}{close_{t-1}}\right|,\qquad q_t=\frac{m_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(m_t-\mu_{t-1}),\qquad \sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(m_t-\mu_{t-1})^2)
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolatilityBreakoutDetector` 中实现。

## 参考资料

- [ChartSchool：标准差与波动率](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
