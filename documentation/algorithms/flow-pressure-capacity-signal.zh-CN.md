# 订单流压力—容量信号（FlowPressureCapacitySignal）

## 摘要

`FlowPressureCapacitySignal` 是一种按事件时间运行的流式 L1 市场微观结构信号。它先衡量近期买方或卖方发起的成交订单流，相对于其前方盘口显示流动性的规模有多大，再判断这些流动性是被**消耗、补充，还是撤走**。

关键就在于这种区分。原始订单簿失衡无法分辨真正的流动性真空与只闪现一次更新的大额队列；原始有符号订单流也无法分辨卖单队列正在被吃掉，还是每次遭到买入后都会重新补单。该信号以因果方式结合报价流与成交流：

1. 按事件时间滤波的 L1 队列失衡；
2. 主动买卖订单流除以对手方近触价容量；
3. 成交后同价位补单的推断值；以及
4. 将买卖盘撤单与价位耗尽视作流动性脆弱性。

该实现是 RTTA 的工程综合，并非逐字复现某一篇论文中的估计器。其主要实证动机来自 Chang（2026）：相较原始订单流，方向性订单流与近触价容量之比对短期收益率和逆向选择风险更有信息量。实现也吸收了 Nittur Anantha、Jain 与 Maiti（2025）的实时结构滤波结论：瞬时订单簿活动会削弱原始失衡的有效性，而成交事件失衡与未来价格变动具有更强的因果对齐关系。

## 更新 API

精简 API 接受净有符号主动成交量：

```python
indicator = rtta.FlowPressureCapacitySignal()

out = indicator.update(
    bid_price,
    bid_size,
    ask_price,
    ask_size,
    signed_trade_volume,  # 买入为正，卖出为负
)
```

若两次报价更新之间既有买入又有卖出，六输入重载不会丢失信息，应优先使用：

```python
out = indicator.update(
    bid_price,
    bid_size,
    ask_price,
    ask_size,
    buy_volume,
    sell_volume,
)
```

`buy_volume` 和 `sell_volume` 是在**上一个已接受报价之后、且不晚于当前报价**发生的主动成交量。报价数量与成交量必须使用相同单位。

常规 RTTA 状态 API 均可用：

```python
indicator.advance(bid, bid_size, ask, ask_size, signed_volume)
out = indicator.last()
indicator.reset()

batch = indicator.batch(
    bid_prices, bid_sizes, ask_prices, ask_sizes, signed_volumes
)

detailed_batch = indicator.batch(
    bid_prices, bid_sizes, ask_prices, ask_sizes, buy_volumes, sell_volumes
)
```

批量接口支持 NumPy `float32` 和 `float64`。Pandas 表格批处理使用 `bid_price`、`bid_size`、`ask_price`、`ask_size` 和 `signed_volume` 列。

## 构造函数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `half_life_updates` | `32.0` | 队列、订单流、补单和撤单状态按事件数计算的半衰期。 |
| `queue_weight` | `0.25` | 持续性显示队列失衡的权重。 |
| `pressure_weight` | `1.0` | 主动订单流相对于对手方容量的权重。 |
| `replenishment_weight` | `0.75` | 有符号吸收的权重：卖盘补单抵抗买入，买盘补单抵抗卖出。 |
| `fragility_weight` | `0.25` | 卖盘撤单减去买盘撤单的权重。 |
| `score_scale` | `1.0` | 在最终 `tanh` 饱和之前应用的缩放系数。 |
| `entry_threshold` | `0.35` | 离开空仓状态所需的分数绝对值。 |
| `exit_threshold` | `0.15` | 返回空仓状态的迟滞阈值。 |
| `warmup` | `8` | `signal` 可以启用之前所需的有效报价更新数。 |
| `fillna` | `True` | 预热期间返回诊断值（`False` 则返回 NaN）。 |

## 输出

| 字段 | 含义 |
|---|---|
| `signal` | 带进出场迟滞的离散方向，取值为 `{-1, 0, +1}`。 |
| `score` | `[-1, 1]` 范围内的连续复合压力分数。 |
| `fair_value` | 中间价加半个价差乘以 `score`；始终位于当前买卖报价之内。 |
| `microprice` | 标准的 L1 数量加权微价格。 |
| `raw_queue_imbalance` | 瞬时 `(bid_size - ask_size) / total_size`。 |
| `queue_imbalance` | 按事件时间作指数滤波的队列失衡。 |
| `flow_imbalance` | 衰减后的主动成交 `(buy - sell) / (buy + sell)` 订单流。 |
| `pressure` | 主动订单流与对手方容量之比的有符号对数。 |
| `replenishment` | 有符号吸收通道；正值表示买盘支撑，负值表示卖盘供给。 |
| `fragility` | 卖方撤单减去买方撤单；正值看涨。 |
| `spread_bps` | 当前报价价差相对于中间价的基点数。 |

`fair_value` 是一个有界诊断量，并不声称穿越价差后仍有利润。应像 Massive 示例那样，用未来中间价验证 `score`，并以延迟后的买卖报价回测策略。

## 工作原理

### 队列失衡与微价格

对于最优买卖价 \(P^b_t,P^a_t\) 和显示数量 \(Q^b_t,Q^a_t\)，

\[
I_t = \frac{Q^b_t-Q^a_t}{Q^b_t+Q^a_t}, \qquad
M_t = \frac{P^b_t+P^a_t}{2}.
\]

常见的数量加权中间价为：

\[
\operatorname{micro}_t =
\frac{P^a_t Q^b_t + P^b_t Q^a_t}{Q^b_t+Q^a_t}
= M_t + \frac{P^a_t-P^b_t}{2} I_t.
\]

Stoikov 的微价格将公允价格形式化为以价差和失衡为条件、最终中间价的期望。近期高分辨率研究同样从这种 L1 位移出发，再使用订单簿动态进行修正。本指标有意采用幅度较小、透明且完全在线的动态修正。

为了避免单次更新的数量闪烁占据主导地位，RTTA 按事件时间对失衡作滤波。设半衰期为 \(H\)，\(d=2^{-1/H}\)：

\[
\bar I_t = d\bar I_{t-1} + (1-d)I_t.
\]

### 经成交调整的队列核算

假设两次报价快照之间卖价不变，期间发生了主动买入量 \(V^+_t\)。忽略隐藏流动性，可见队列恒等式为：

\[
Q^a_t = Q^a_{t-1} - V^+_t + A^a_t - C^a_t,
\]

其中 \(A\) 是新增流动性，\(C\) 是撤销流动性。L1 数据只能辨识二者的净值，因此指标记录：

\[
R^a_t=\max(Q^a_t-Q^a_{t-1}+V^+_t,0),\qquad
W^a_t=\max(Q^a_{t-1}-V^+_t-Q^a_t,0).
\]

买方一侧的方程将 \(V^+\) 替换为主动卖出量 \(V^-\)。当卖价上移或买价下移时，前一队列被视为已撤走或耗尽；当卖价或买价改善时，则视作在触价处新增补单。

这就是为什么必须正确合并连续报价**之间**的成交。如果把一笔成交附到未来报价上，该例程会把成交误认为补单，并造成信息泄漏。

### 衰减订单流与容量

主动订单流、补单和撤单均使用指数型脉冲噪声状态，而不是矩形窗口：

\[
\bar X_t = d\bar X_{t-1}+X_t.
\]

有效对手方容量为：

\[
C^a_t=Q^a_t+\bar R^a_t,\qquad C^b_t=Q^b_t+\bar R^b_t.
\]

压力通道在对称、有界的尺度上，比较买方压力与卖盘容量以及卖方压力与买盘容量：

\[
P_t=\tanh\!\left[
\log\!\left(1+\frac{\bar V^+_t}{C^a_t}\right)
-\log\!\left(1+\frac{\bar V^-_t}{C^b_t}\right)
\right].
\]

因此，100 股买单冲击只有 10 股的卖盘，远比同样的订单流面对 10,000 股显示卖盘更重要。

### 吸收与脆弱性

补单提供了此前缺失的判别维度。令：

\[
A^a_t=\frac{\bar R^a_t}{\bar R^a_t+\bar V^+_t},\qquad
A^b_t=\frac{\bar R^b_t}{\bar R^b_t+\bar V^-_t}.
\]

按买卖双方各自在近期主动订单流中的占比加权，可得：

\[
R_t = w^-_t A^b_t - w^+_t A^a_t.
\]

因此，买盘面对卖出而反复补单表示看涨支撑；卖盘面对买入而反复补单表示看跌供给。卖盘完全补充时，可以抵消原本看涨的买入压力，而不是盲目跟随该压力。

撤单脆弱性为：

\[
F_t =
\frac{\bar W^a_t}{\bar W^a_t+Q^a_t+\bar R^a_t}
-
\frac{\bar W^b_t}{\bar W^b_t+Q^b_t+\bar R^b_t}.
\]

卖盘消失表示看涨；买盘消失表示看跌。

### 复合分数与信号

使用构造函数权重 \(w_I,w_P,w_R,w_F\)：

\[
S_t=\tanh\left\{s\left(
w_I\bar I_t+w_P P_t+w_R R_t+w_F F_t
\right)\right\}.
\]

`score` 即 \(S_t\)，并且：

\[
\operatorname{fair}_t=M_t+\tfrac12(P^a_t-P^b_t)S_t.
\]

`signal` 在达到 `entry_threshold` 时启用，并持续保持该方向，直到 `score` 穿越 `exit_threshold`；这样可以避免仓位随每次报价更新频繁翻转。

每次更新都具有因果性，构造完成后不再分配内存，时间复杂度为 \(O(1)\)。

## Massive/Polygon 事件对齐

参考示例按 SIP 时间戳合并 `StockQuoteDatabase` 与 `StockTradeDatabase`。每笔成交都根据该时间戳时已知的最近报价进行分类；买入量和卖出量累积到下一次报价，再通过详细重载传入。示例还会：

- 跳过锁盘、交叉盘和零值报价；
- 在每个常规交易时段开始时重置状态；
- 用较晚的中间价评估抽样预测；
- 报告分数/价格变动相关性和方向性优势；以及
- 以延迟后的买卖报价进行模拟交易，从而显式支付价差。

报价数量和成交量必须使用相同单位。[Massive 从 2025 年 11 月 3 日起以股数而不是整手报告股票报价数量](https://www.massive.com/changelog#stock-quote-size-reporting-change)，随后也重新生成了历史报价平面文件。因此，当前文件应使用示例默认的 `--quote-size-multiplier 1`；只有仍以整手保存报价数量的旧缓存文件才应将其设为 `100`。

参见 `examples/flow_pressure_capacity_from_massive_speedup.py`。

## 局限性

- 合并后的 L1 数量不是按订单 ID 维护的队列。新增与撤单是净额推断，而且无法观测隐藏单/冰山单流动性。
- SIP 成交与报价流在时间戳相同时，事件顺序可能含糊。只有当两种数据流共享可靠、可验证的时钟或事件排序时，才应使用参与者时间戳。
- 成交条件、修正、零股交易和场外成交需要生产级的纳入规则。示例只应用最低限度的过滤。
- 事件时间半衰期能适应活跃度，但在不同标的和市场状态下代表不同的真实时间长度。如果要求时钟时间上的可比性，应先进行降采样。
- 支撑压力/容量思想的实证证据包含加密资产和非美国市场数据。投入资金前，必须在目标美国股票样本空间上进行样本外验证。

## 参考资料

- Lawrence Chang，[《Do Order-Book States Predict Passive-Buy Toxicity? Evidence from BTC Perpetual Futures》](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6693260)，2026。
- Aditya Nittur Anantha、Shashi Jain 与 Prithwish Maiti，[《Order Book Filtration and Directional Signal Extraction at High Frequency》](https://arxiv.org/abs/2507.22712)，2025。
- Christian D. Blakely，[《High Resolution Microprice Estimates from Limit Orderbook Data Using Hyperdimensional Vector Tsetlin Machines》](https://arxiv.org/abs/2411.13594)，2024。
- Sasha Stoikov，[《The Micro-Price: A High Frequency Estimator of Future Prices》](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)，2018。
- Rama Cont、Arseniy Kukanov 与 Sasha Stoikov，[《The Price Impact of Order Book Events》](https://arxiv.org/abs/1011.6402)，2014。
- Yang Zhou、Jianwen Chen 与 Ruipeng Wei，[《Order Splitting and Liquidity Replenishment Are Jointly Necessary for the Square-Root Law of Market Impact》](https://arxiv.org/abs/2607.04280)，2026。
