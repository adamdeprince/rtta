# CointegrationBreakdownMonitor

## 摘要

`CointegrationBreakdownMonitor` 用 EWMA 对冲比率估计和流式残差 z 分数监控配对关系失效。

## 更新 API

```python
result = rtta.CointegrationBreakdownMonitor().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次更新先估计两个序列的在线对冲比率和截距，再把当前残差相对历史残差均值与方差标准化。绝对 z 分数通过回滞逻辑转换为持续的关系失效标志。

## 递推公式

令 \(z_t = (real0_t, real1_t)\) 为一次更新接收的观测。

\[
\beta_t=\frac{C^{xy}_t}{V^y_t}, \qquad e_t=x_t-(\beta_t y_t+\alpha_t)
\]

\[
q_t=\left|\frac{e_t-\bar{e}_{t-1}}{\sqrt{\max(s^2_{e,t-1},\epsilon)}}\right|
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CointegrationBreakdownMonitor` 中实现。

## 参考资料

- [背景资料：协整](https://en.wikipedia.org/wiki/Cointegration)
