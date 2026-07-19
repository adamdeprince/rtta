# LiquidityRegimeDetector

## 摘要

`LiquidityRegimeDetector` 用单位美元成交量的绝对收益率构造 EWMA Amihud 式流动性状态。

## 更新 API

```python
result = rtta.LiquidityRegimeDetector().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器先计算单位成交金额对应的绝对收益，对其做 EWMA 平滑，再应用双向进入/退出回滞。

## 递推公式

\[
DV_t=|close_t|\max(volume_t,0),\qquad I_t=\frac{|(close_t-close_{t-1})/close_{t-1}|}{\max(DV_t,\epsilon)}
\]

\[
q_t=\alpha I_t+(1-\alpha)q_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class LiquidityRegimeDetector` 中实现。

## 参考资料

- [Amihud 非流动性估计量](https://ba-odegaard.no/teach/notes/liquidity_estimators/amihud_estimator/amihud_lectures.pdf)
