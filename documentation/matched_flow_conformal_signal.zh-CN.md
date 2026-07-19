# MatchedFlowConformalSignal

`MatchedFlowConformalSignal` 是一种日内 OHLCV 信号。它只在方向性价格变动、带符号成交量流、价格相对 VWAP 的位置以及相对活跃度指向同一方向，而且预期变动大于近期经验误差区间时，才尝试进行交易。

这是一个组合式研究原型，并非对某一篇论文的直接复现。主要参考资料包括：

- Chordia、Roll 与 Subrahmanyam，"Order imbalance, liquidity, and market returns"，《Journal of Financial Economics》，2002。论文将订单失衡作为交易活跃度指标，研究其与流动性和市场收益之间的关系。公开元数据：https://authors.library.caltech.edu/95132/
- Chordia 与 Subrahmanyam，"Order imbalance and individual stock returns: Theory and evidence"，《Journal of Financial Economics》，2004。论文研究具有自相关性的订单失衡如何给个股带来价格压力。SSRN 页面：https://papers.ssrn.com/sol3/papers.cfm?abstract_id=354122
- Xu 与 Xie，"Sequential Predictive Conformal Inference for Time Series"，arXiv:2212.03463，2022。论文阐述了为何要针对非可交换时间序列，对非一致性分数进行序贯校准。arXiv/HF 页面：https://huggingface.co/papers/2212.03463

RTTA 的实现有意采用比这些论文更简单的方案。它不会根据带符号成交或报价来估计真正的订单失衡，而是把相邻 K 线收盘收益的符号作为带符号流量的廉价代理，再按美元成交量进行缩放。

## API

```python
rtta.MatchedFlowConformalSignal(
    horizon_bars=12,
    calibration_window=250,
    calibration_quantile=0.80,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.05,
    participation_cap=0.02,
    fillna=False,
)
```

`update(open, high, low, close, volume, normal_dollar_volume=nan, market_cap=nan, reset_session=False)` 每次接收一根聚合 K 线。示例脚本使用 Massive/Polygon 的股票成交聚合数据，但任何间隔规则的 OHLCV K 线流都可以使用。

## 状态

指标维护以下状态：

- 用于计算 3、6 和 12 根 K 线动量的收盘价滞后值；
- 6 根和 12 根 K 线的滚动收益波动率；
- 3、6 和 12 根 K 线窗口内 `alpha_flow` 的滚动和；
- 3 根和 6 根 K 线窗口内参与流量的滚动和；
- 根据 K 线 VWAP 或收盘价乘成交量构建的日内时段 VWAP；
- 正常美元成交量的 EWMA 后备估计；
- 固定大小的待验证预测队列；
- 已实现预测误差的滚动分位数。

`reset_session=True` 会清除收盘价滞后、流量窗口、待验证预测和时段 VWAP 等日内状态，但不会清除寿命更长的正常美元成交量 EWMA，也不会清除校准残差分位数。

## 特征构造

对每一根有效 K 线：

1. 计算 `dollar_volume = close * max(volume, 0)`。
2. `normal_dollar_volume` 依次取调用者提供的数值、内部 EWMA 或当前美元成交量中的可用值。
3. 计算相邻收盘价的对数收益 `ret1`。
4. 以 `sign(ret1)` 作为带符号流量的方向。
5. 根据累计价格成交量和累计成交量更新时段 VWAP。
6. 计算：
   - `vwap_gap = close / session_vwap - 1`
   - `rel_dollar_volume = dollar_volume / normal_dollar_volume`
   - 若提供市值，则 `alpha_flow = sign(ret1) * dollar_volume / market_cap`；否则按正常美元成交量缩放
   - `participation = sign(ret1) * dollar_volume / normal_dollar_volume`
7. 将 `alpha_flow` 和 `participation` 限制在 `[-10, 10]` 范围内。
8. 把这些数值推入各滚动流量窗口。

动量是 3、6 和 12 根 K 线对数收益的加权组合：

```text
momentum = 0.20 * ret_3 + 0.35 * ret_6 + 0.45 * ret_12
```

`flow_score` 通过 `tanh` 压缩累计带符号流量；是否提供 `market_cap` 会决定所用的归一化方式：

```text
flow_score = tanh(alpha_flow_12_sum / alpha_norm
                 + 0.50 * participation_6_sum / 6)
```

相对活跃度也会被压缩：

```text
activity_score = tanh((rel_dollar_volume - 1) / 2)
```

最高价与最低价的区间会抑制价差较宽或噪声较大的 K 线上的预测：

```text
spread_proxy = max(0, high - low) / close
spread_dampen = 1 / (1 + 25 * spread_proxy)
```

## 预测

对未来一个预测期限的原始对数收益预测为：

```text
raw_prediction =
    0.35   * momentum
  + 0.0010 * flow_score
  + 0.05   * vwap_gap
  + 0.0005 * activity_score

prediction = spread_dampen * raw_prediction
```

这套公式有意采用启发式设计。指标的目的不是在内部拟合模型，而是公开一个稳定的增量信号，并且只在多项日内条件同时成立时作出反应。

## 误差区间

每次预测都会与入场时的收盘价一起保存。经过 `horizon_bars` 次更新后，指标会用已实现对数收益与保存的预测进行比较：

```text
realized = log(current_close / old_close)
error = abs(realized - old_prediction)
```

误差会被推入滚动分位数。积累足够多的误差后，`radius` 为：

```text
radius = max(rolling_error_quantile, cost_buffer)
```

校准尚未就绪时，使用以下后备值：

```text
radius = max(2 * cost_buffer,
             max(0.0007, 1.25 * volatility_12 * sqrt(horizon_bars)))
```

这种做法受到共形推断的启发，但不提供正式的有限样本共形保证；它只是一个在线经验误差区间。

## 交易输出

分数的计算方式为：

```text
score = prediction / (radius + cost_buffer)
```

信号为：

```text
signal = +1 if prediction > entry_z * (radius + cost_buffer)
signal = -1 if prediction < -entry_z * (radius + cost_buffer)
signal =  0 otherwise
```

目标仓位比例设有上限：

```text
target_fraction = max_abs_target_fraction * clamp(score / 3, -1, 1)
```

`max_trade_dollars = participation_cap * normal_dollar_volume` 为外部执行层提供流动性上限。

## 输出

- `prediction`：未来一个预测期限的预期对数收益。
- `radius`：经验不确定性/误差区间半径。
- `score`：预测值除以误差半径与成本缓冲之和。
- `signal`：`-1`、`0` 或 `+1`。
- `target_fraction`：受上限约束的建议敞口。
- `alpha_flow`：按市值或正常美元成交量缩放的带符号美元成交量。
- `participation`：按正常美元成交量缩放的带符号美元成交量。
- `flow_score`：经压缩的近期带符号流量指标。
- `momentum`：多 K 线周期加权动量。
- `volatility`：12 根 K 线收益率的滚动波动率。
- `vwap_gap`：收盘价相对时段 VWAP 的位置。
- `rel_dollar_volume`：当前美元成交量相对正常水平的比例。
- `max_trade_dollars`：流动性上限。
- `realized_error`：最近一条已经到期的预测误差（如有）。

## 预期用途

本指标适合用于流动性较好的日内 K 线，通常为 1 至 5 分钟周期，并应在每个交易时段边界重置。它最适合作为更大系统中的排序或准入特征；价差、费用、库存、敞口和执行仍应由该系统的其他部分处理。
