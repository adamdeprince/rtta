# ExecutionCostSlippageRegimeDetector

## 摘要

`ExecutionCostSlippageRegimeDetector` 根据成交价相对报价中点的偏离，检测执行成本与滑点状态。

## 更新 API

```python
result = rtta.ExecutionCostSlippageRegimeDetector().update(trade_price, bid_price, ask_price)
```

`update(...)` 每次接收成交价、买价和卖价；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器先计算成交价偏离报价中点的相对幅度，再通过进入/退出回滞生成稳定状态。

## 递推公式

\[
mid_t=\frac{bid_t+ask_t}{2},\qquad q_t=\frac{|trade_t-mid_t|}{\max(|mid_t|,\epsilon)}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ExecutionCostSlippageRegimeDetector` 中实现。

## 参考资料

- [背景资料](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
