# 分解订单流失衡（DecomposedOrderFlowImbalance）

## 摘要

`DecomposedOrderFlowImbalance` 将 Cont 风格的最优报价压力拆分为三个可加通道——**新增**、**撤单**和**成交**（代理量）；每个通道均使用滚动求和窗口，并以三者之和作为 `total`。输入仅为盘口最优档的买卖价格与数量。

## 更新 API

```python
import rtta

ind = rtta.DecomposedOrderFlowImbalance(window=1, fillna=True)
result = ind.update(bid_price, bid_size, ask_price, ask_size)
# result.add, result.cancel, result.trade, result.total
```

`advance(...)` 更新状态但不返回结果。批量辅助函数接受并行的买卖价格与数量数组。

## 工作原理

标准 Cont OFI 把流动性新增、撤单和主动成交混合为一个标量。本指标将每次报价变动归入：

- **新增（Add）**——增加 Cont 风格买方压力的新流动性（买价改善或数量增加；按照 Cont 的符号约定，也包括卖价变差）。
- **撤单（Cancel）**——被移除的流动性，其 Cont 符号与新增相反。
- **成交（Trade）**——启发式代理：中间价上行，同时卖方数量减少且卖价没有改善（买方吃掉卖单）；或中间价下行，同时买方数量减少且买价没有变差（卖方打掉买单）。

每个分量都在各自的 `RollingSumWindow` 中累积。首个快照只用于初始化前一状态，各分量贡献均为零。

## 递推公式

令 \((b_t,B_t,a_t,A_t)\) 为盘口最优买价、买量、卖价和卖量，并令 \(m_t=\tfrac12(b_t+a_t)\)。首个时点的瞬时贡献为零。此后，瞬时分量 \(e^{\mathrm{add}}_t\)、\(e^{\mathrm{cancel}}_t\)、\(e^{\mathrm{trade}}_t\) 按如下规则计算。

**买方一侧**

\[
\begin{aligned}
b_t > b_{t-1} &\Rightarrow e^{\mathrm{add}} {+}{=} B_t,\\
b_t < b_{t-1} &\Rightarrow e^{\mathrm{cancel}} {-}{=} B_{t-1},\\
b_t = b_{t-1},\; B_t > B_{t-1} &\Rightarrow e^{\mathrm{add}} {+}{=} B_t - B_{t-1},\\
b_t = b_{t-1},\; B_t < B_{t-1} &\Rightarrow e^{\mathrm{cancel}} {-}{=} B_{t-1} - B_t.
\end{aligned}
\]

**卖方一侧**（Cont 符号与买方相反）

\[
\begin{aligned}
a_t < a_{t-1} &\Rightarrow e^{\mathrm{add}} {-}{=} A_t,\\
a_t > a_{t-1} &\Rightarrow e^{\mathrm{cancel}} {+}{=} A_{t-1},\\
a_t = a_{t-1},\; A_t > A_{t-1} &\Rightarrow e^{\mathrm{add}} {-}{=} A_t - A_{t-1},\\
a_t = a_{t-1},\; A_t < A_{t-1} &\Rightarrow e^{\mathrm{cancel}} {+}{=} A_{t-1} - A_t.
\end{aligned}
\]

**成交代理量**

\[
\begin{aligned}
m_t > m_{t-1} \;\land\; A_t < A_{t-1} \;\land\; a_t \le a_{t-1}
&\Rightarrow e^{\mathrm{trade}} {+}{=} A_{t-1} - A_t,\\
m_t < m_{t-1} \;\land\; B_t < B_{t-1} \;\land\; b_t \ge b_{t-1}
&\Rightarrow e^{\mathrm{trade}} {-}{=} B_{t-1} - B_t.
\end{aligned}
\]

滚动输出（\(W=\)`window`）：

\[
\mathrm{add}_t = \sum e^{\mathrm{add}},\quad
\mathrm{cancel}_t = \sum e^{\mathrm{cancel}},\quad
\mathrm{trade}_t = \sum e^{\mathrm{trade}}
\quad\text{（最近 \(W\) 个样本）},
\]

\[
\mathrm{total}_t = \mathrm{add}_t + \mathrm{cancel}_t + \mathrm{trade}_t.
\]

若 `fillna=False` 且任一窗口尚未填满，四个字段均为 `NaN`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class DecomposedOrderFlowImbalance` 中实现。新增、撤单和成交分别保存在三个独立的 `RollingSumWindow` 实例中。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》（arXiv:1011.6402）](https://arxiv.org/abs/1011.6402)
