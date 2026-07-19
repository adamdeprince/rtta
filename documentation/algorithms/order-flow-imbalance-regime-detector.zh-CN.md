# OrderFlowImbalanceRegimeDetector

## 摘要

`OrderFlowImbalanceRegimeDetector` 对订单流失衡做 EWMA 平滑，并用回滞识别买压或卖压状态。

## 更新 API

```python
result = rtta.OrderFlowImbalanceRegimeDetector().update(bid_price, bid_size, ask_price, ask_size)
```

`update(...)` 每次接收最优买卖价及其挂单量；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器根据最优买卖价和挂单量的逐笔变化计算订单流失衡，以当前盘口总深度归一化，再做 EWMA 平滑并应用双向回滞。

## 递推公式

\[
e_t=\mathbf1[bid_t\ge bid_{t-1}]bidSize_t-\mathbf1[bid_t\le bid_{t-1}]bidSize_{t-1}-\mathbf1[ask_t\le ask_{t-1}]askSize_t+\mathbf1[ask_t\ge ask_{t-1}]askSize_{t-1}
\]

\[
n_t=\frac{e_t}{\max(bidSize_t+askSize_t,\epsilon)},\qquad q_t=\alpha n_t+(1-\alpha)q_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class OrderFlowImbalanceRegimeDetector` 中实现。

## 参考资料

- [订单流失衡论文](https://arxiv.org/abs/1011.6402)
