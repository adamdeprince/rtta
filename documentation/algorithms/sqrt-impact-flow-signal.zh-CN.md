# 平方根冲击订单流信号（SqrtImpactFlowSignal）

## 摘要

`SqrtImpactFlowSignal` 是一种由**平方根市场冲击**定律和 Cont 风格有符号订单流构建的流式研究信号。它面向 Massive/Polygon 的**聚合 K 线**（收盘价、成交量、可选 VWAP），也可以在自行聚合逐笔数据时接收**由逐笔计算的有符号成交额**。

其经济逻辑为：

1. 元订单的平均价格冲击按 \(Y\,\sigma\sqrt{Q/V}\) 缩放（Tóth 等；Bouchaud 的综述；2025 年关于冲击、失衡和波动率的研究）。
2. 若某根 K 线的有符号成交量所隐含的冲击，大于实际实现的收益率，则冲击“尚未完成” → 沿订单流方向**延续**。
3. 若收益率超过成交量隐含冲击 → **临时冲击回归**。
4. 可选的 **VWAP 距离**把成交价格压力与 K 线的成交量加权位置（Polygon 聚合字段 `vw`）对齐。

## 更新 API

```python
# 聚合路径（由收盘到收盘逐笔规则确定订单流）
out = rtta.SqrtImpactFlowSignal().update(close, volume)
# out.signal, out.score, out.impact, out.residual, out.continuation,
# out.reversion, out.participation, out.flow, out.volatility, out.vwap_gap

# 可选：由逐笔数据计算的真实有符号成交额 + K 线 VWAP
out = rtta.SqrtImpactFlowSignal().update(
    close, volume, signed_dollar_volume=signed_dv, vwap=bar_vwap
)

# 与 OHLCV 兼容的重载（忽略 open/high/low）
out = rtta.SqrtImpactFlowSignal().update(open, high, low, close, volume)

batch = rtta.SqrtImpactFlowSignal().batch(close, volume)
```

构造参数：

| 参数 | 默认值 | 作用 |
|----------|---------|------|
| `impact_coefficient` | `1.0` | \(I=Y\sigma\sqrt{Q/V}\) 中的 \(Y\) |
| `adv_span` | `50` | 平均成交额 \(V\) 的 EWMA 跨度 |
| `vol_span` | `20` | 收益率尺度 \(\sigma\) 的 EWMA 跨度 |
| `continuation_weight` | `1.0` | 未完成冲击的权重 |
| `reversion_weight` | `0.5` | 超调回归的权重 |
| `vwap_weight` | `0.25` | VWAP 距离对齐项的权重 |
| `entry_z` / `exit_z` | `0.75` / `0.25` | 离散 `signal` 的迟滞阈值 |
| `fillna` | `True` | 预热期间返回 `0` 还是 `NaN` |

## 工作原理

### 平方根冲击

从实证与理论上看，成交量 \(Q\) 相对于近期活动量 \(V\) 所造成的平均中间价/价格绝对位移为：

\[
I_t = Y\,\sigma_t\sqrt{\frac{Q_t}{V_t}}
\]

其中 \(\sigma_t\) 是局部波动率尺度。RTTA 令 \(Q_t=C_tV^{\mathrm{sh}}_t\)（该 K 线成交额），\(V_t\) 则为成交额 EWMA（ADV 代理量）。

### 有符号订单流

对于没有逐笔成交流的聚合数据，订单流符号按收盘价的**逐笔规则**确定：

\[
s_t = \mathrm{sign}(\log C_t - \log C_{t-1}).
\]

若提供由逐笔分类（逐笔规则 / 基于报价的 Lee–Ready）得到的 `signed_dollar_volume`，则 \(s_t=\mathrm{sign}(\mathrm{signed\_dollar})\)。

### 延续与回归

令 \(r_t=\log(C_t/C_{t-1})\)，并令 \(I_t\) 如上。

\[
\begin{aligned}
\mathrm{continuation}_t &= s_t\cdot\frac{\max(0,\, I_t - |r_t|)}{\sigma_t} \\
\mathrm{reversion}_t &= -\mathrm{sign}(r_t)\cdot\frac{\max(0,\, |r_t| - I_t)}{\sigma_t}
\end{aligned}
\]

- **延续** > 0：有符号成交量很大，但价格移动较小 → 预期价格继续沿订单流方向漂移（永久冲击不完整 / 反应不足）。
- **回归** > 0：价格移动大于成交量隐含冲击 → 临时冲击应发生均值回归。

### VWAP 对齐

若 K 线 VWAP \(W_t\) 可用：

\[
g_t = \frac{C_t}{W_t} - 1, \qquad
\mathrm{align}_t = s_t\cdot\frac{g_t}{\sigma_t}.
\]

买入并收在 VWAP 上方（或卖出并收在下方）会强化压力。

### 分数与离散信号

\[
\mathrm{raw}_t = \tanh\bigl(
  w_c\,\mathrm{continuation}_t + w_r\,\mathrm{reversion}_t + w_v\,\mathrm{align}_t
\bigr)
\]

`score` 即 \(\mathrm{raw}_t\)。对 `score` 计算在线 z 分数，再应用迟滞（`entry_z` / `exit_z`），产生离散 `signal` \(\in\{-1,0,+1\}\)。

## 递推过程

状态包括：前一收盘价、ADV EWMA \(V_t\)、作为 \(\sigma_t\) 的 \(|r|\) EWMA、分数 EWMA 均值/方差，以及离散仓位。

1. \(r_t=\log(C_t/C_{t-1})\)，\(Q_t=C_t\cdot\mathrm{volume}_t\)。
2. \(V_t=(1-\alpha_V)V_{t-1}+\alpha_VQ_t\)。
3. \(\sigma_t=(1-\alpha_\sigma)\sigma_{t-1}+\alpha_\sigma|r_t|\)。
4. \(I_t=Y\sigma_t\sqrt{Q_t/V_t}\)，残差为 \(r_t-s_tI_t\)。
5. 构造延续、回归和 VWAP 对齐项 → tanh 分数 → 采用 z 分数迟滞的信号。

所有更新的时间复杂度均为 \(O(1)\)，且具有因果性。

## 交易（买入 / 卖出）

离散交易指令是 **`result.signal`**：

| `signal` | 操作（参见示例） |
|----------|----------------------|
| `+1` | **买入（BUY）** / 持有多头 |
| `-1` | **卖空（SELL）**，若只做多则退出多头 |
| `0` | **空仓（FLAT）** / 退出 |

`score` 可选作置信度或头寸规模。完整模拟交易循环会输出 CSV，其中 `action` 取 `{BUY,SELL,COVER,HOLD,...}`，并支持只做多、`--allow-short` 和 `--min-score`；代码位于：

`examples/sqrt_impact_flow_from_massive_speedup.py`

## 实现说明

在 `src/rtta/indicator.cpp` 的 `class SqrtImpactFlowSignal` 中实现。

这**不是**完整的限价订单簿元订单重构器。它是平方根冲击残差结构的 K 线级 / 可选逐笔有符号代理量；该结构受到近期冲击—失衡研究的强调，并针对 Polygon/Massive 字段作了工程化实现。

## 参考资料

- Tóth 等，《Anomalous price impact and the critical nature of liquidity in limit order books》，*Physical Review X*，2011（平方根冲击现象）。
- Bouchaud、Farmer、Lillo——关于市场冲击和平方根定律的综述。
- [arXiv:2509.05065](https://arxiv.org/abs/2509.05065)——《The Subtle Interplay between Square-root Impact, Order Imbalance & Volatility II》（2025）：广义订单流与收益率之间的相关结构。
- Cont、Kukanov、Stoikov，《The Price Impact of Order Book Events》，[arXiv:1011.6402](https://arxiv.org/abs/1011.6402)（OFI / 有符号订单流基础）。
