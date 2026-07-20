# 消息事件订单流失衡（MessageEventOrderFlowImbalance）

## 摘要

`MessageEventOrderFlowImbalance` 是 RTTA 对消息流订单流失衡的流式实现：它不使用 Cont 风格的最优报价快照差，而是对离散限价订单簿 / 成交事件（新增、撤单、成交）的有符号贡献作滚动求和。

## 更新 API

```python
result = rtta.MessageEventOrderFlowImbalance(window=50, fillna=True).update(
    event_type, side, qty
)
# result.ofi, result.event, result.signed_size
```

- `event_type`：`1` = 新增，`2` = 撤单，`3` = 成交（其他代码退化为只计算有符号数量）
- `side`：`+1` = 买盘 / 主动买方，`-1` = 卖盘 / 主动卖方
- `qty`：非负事件数量（若为负值，则使用绝对值）

`advance(...)` 使用相同的三个输入，但不返回 Python 对象。多输出 `batch(event_type, side, qty)` 分别为 `ofi`、`event` 和 `signed_size` 返回数组。

## 工作原理

快照 OFI（`OrderFlowImbalance`）根据最优买卖价和数量的变化重构压力。消息流 OFI 则直接处理交易所事件流：流动性新增、撤单和成交。RTTA 把每个事件映射为有符号贡献，并在事件的滚动窗口内累积这些贡献。滚动和为正，表示近期消息历史中净买盘/买入压力；为负则表示净卖盘/卖出压力。

## 递推公式

令 \(e_t\in\{1,2,3\}\) 为事件类型，\(s_t\in\{+1,-1\}\) 为一侧，\(q_t\ge0\) 为数量。定义贡献：

\[
\delta_t =
\begin{cases}
+s_t q_t & e_t = 1 \quad\text{（新增）} \\
-s_t q_t & e_t = 2 \quad\text{（撤单）} \\
+s_t q_t & e_t = 3 \quad\text{（成交）} \\
+s_t q_t & \text{其他情况}
\end{cases}
\]

在由最近 \(n\) 个贡献组成的滚动窗口 \(W_t\)（`window`）上：

\[
\operatorname{OFI}_t = \sum_{\tau\in W_t} \delta_\tau
\]

每次更新还会把最新贡献作为 `event` 输出，并把有符号数量 \(s_tq_t\) 作为 `signed_size` 输出。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class MessageEventOrderFlowImbalance` 中实现，使用 `RollingSumWindow`。当 `fillna=False` 时，在窗口填满之前，`ofi` 为 `NaN`。

这是面向研究的事件 API；若不先转换数据源，就不能与盘口最优档 `OrderFlowImbalance` 互换使用。

## 参考资料

- [Cont、Kukanov 与 Stoikov，《The Price Impact of Order Book Events》，arXiv:1011.6402](https://arxiv.org/abs/1011.6402)（Cont 风格 OFI 的动机；RTTA 的快照形式为 `OrderFlowImbalance`，而本类把同一失衡思想应用于显式消息事件。）
- [Cont、Cucuringu 等，多档 / 限价订单簿订单流文献综述](https://arxiv.org/abs/2104.14067)
