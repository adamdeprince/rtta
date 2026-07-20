# 多档订单流失衡（MultiLevelOrderFlowImbalance）

## 摘要

`MultiLevelOrderFlowImbalance` 是 RTTA 对多档深度 Cont 风格订单流失衡的流式实现。在限价订单簿每个档位 \(\ell=0,\ldots,L-1\)，连续的盘口快照会产生一个 Cont 事件贡献；这些事件通过固定长度窗口滚动求和，再汇总为 `total`、等权 `mean`，以及前五档序列 `l1`…`l5`。

## 更新 API

```python
import numpy as np
import rtta

ind = rtta.MultiLevelOrderFlowImbalance(levels=5, window=1, fillna=True)
# 每个时点：长度为 `levels` 的深度向量
result = ind.update(bid_price, bid_size, ask_price, ask_size)
# result.total, result.mean, result.l1 ... result.l5

# 批量：时间 × 深度矩阵，形状为 (n_samples, levels)
batch = ind.batch(bid_prices, bid_sizes, ask_prices, ask_sizes)
```

每次调用 `update(...)` 处理一个深度快照（四个长度为 `levels` 的向量）。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。接受 float32 和 float64 数组。

## 工作原理

Cont、Kukanov 和 Stoikov（以及后来的多档扩展）根据报价修订定义订单流失衡：买卖价格发生跳变时新增/移除的数量，或价格不变时发生的数量变化。RTTA 在订单簿每个档位独立应用该事件映射，保存各档此前的价格/数量，以 `RollingSumWindow(window)` 滚动累积每档事件，并报告：

- `total`——所有档位滚动事件的总和
- `mean`——`total / levels`
- `l1`…`l5`——前五档的滚动贡献（若实际档位更少，缺失档位为零）

第一个快照的贡献为零（没有此前报价）。当 `fillna=False` 时，未填满的滚动窗口在该档输出 NaN，并且不计入 `total`。

## 递推公式

令 \(L\) 为 `levels`，\(W\) 为 `window`；时刻 \(t\) 的档位 \(\ell\) 上，令 \((b^{(\ell)}_t,B^{(\ell)}_t,a^{(\ell)}_t,A^{(\ell)}_t)\) 分别为买价、买量、卖价和卖量。

档位 \(\ell\) 的 Cont 事件（首个观测之后）为：

\[
e^{(\ell)}_t
=
\mathbf{1}\{b^{(\ell)}_t \ge b^{(\ell)}_{t-1}\} B^{(\ell)}_t
-
\mathbf{1}\{b^{(\ell)}_t \le b^{(\ell)}_{t-1}\} B^{(\ell)}_{t-1}
-
\mathbf{1}\{a^{(\ell)}_t \le a^{(\ell)}_{t-1}\} A^{(\ell)}_t
+
\mathbf{1}\{a^{(\ell)}_t \ge a^{(\ell)}_{t-1}\} A^{(\ell)}_{t-1}.
\]

首个时点取 \(e^{(\ell)}_t=0\)。令 \(S^{(\ell)}_t\) 为档位 \(\ell\) 上最近 \(\min(t,W)\) 个事件的滚动和：

\[
S^{(\ell)}_t = \sum_{k=0}^{\min(t,W)-1} e^{(\ell)}_{t-k}
\quad\text{（若 `fillna=False` 且窗口尚未填满，则为 NaN）。}
\]

汇总值：

\[
\text{total}_t = \sum_{\ell=0}^{L-1} S^{(\ell)}_t,\qquad
\text{mean}_t = \frac{\text{total}_t}{L},\qquad
\mathrm{l}_{j,t} = S^{(j-1)}_t \quad (j=1,\ldots,5).
\]

每个时点处理后，把当前 \((b,B,a,A)\) 保存为下次事件所用的此前状态。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class MultiLevelOrderFlowImbalance` 中实现（辅助函数 `cont_ofi_event`）。批量路径要求形状为 `(n_samples, levels)` 的 C 连续矩阵。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》（arXiv:1011.6402）](https://arxiv.org/abs/1011.6402)
