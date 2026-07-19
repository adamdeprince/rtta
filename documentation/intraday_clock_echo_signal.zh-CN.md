# IntradayClockEchoSignal

`IntradayClockEchoSignal` 是一种基于日内相同时刻周期性的信号。它学习一天中哪些时刻在历史上呈现正或负的残差收益，并利用这种时钟模式预测未来 `horizon_bars` 根 K 线的收益。

主要参考论文为：

- Heston、Korajczyk 与 Sadka，"Intraday Patterns in the Cross-Section of Stock Returns"，《Journal of Finance》，2010。论文记录了间隔恰好为整数个交易日的日内收益延续现象，而且该效应会持续多个交易日。SSRN 页面：https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1107590
- Haendler、Heston、Korajczyk 与 Sadka，"The Intra-Day Stock Return Periodicity Puzzle"，SSRN，2025。论文在样本外重新检验该效应，并研究包括类似 VWAP 的交易和收盘市价单交易在内的可能解释。SSRN 页面：https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5749704

RTTA 的实现并不是完整的横截面复现，而是一个按交易品种独立运行的在线指标。它可以从过去的 K 线或显式给出的训练日数据中，学习每个时段槽位的残差收益行为。

## API

```python
rtta.IntradayClockEchoSignal(
    slots_per_session=195,
    horizon_bars=15,
    lookback_days=40,
    min_slot_samples=10,
    calibration_window=500,
    calibration_quantile=0.80,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.03,
    participation_cap=0.02,
    allow_short=True,
    fillna=False,
)
```

对于 390 分钟的美股交易时段，若使用 2 分钟 K 线，应设置 `slots_per_session=195`；若使用 5 分钟 K 线，则设为 78。示例脚本会根据时间间隔计算该数值，除非调用者明确提供。

`update(open, high, low, close, volume, vwap=nan, transactions=nan, market_return=0.0, normal_dollar_volume=nan, slot=0, reset_session=False)` 每次接收一根 K 线。

`train(days)` 方法接收一个交易日记录序列。每个交易日都是由类字典或类元组记录组成的可迭代对象；记录至少应包含开盘价、最高价、最低价、收盘价和成交量，还可选择包含 vwap、transactions、market_return、normal_dollar_volume 和 slot。

## “时钟回声”的含义

论文中的结果关注日内时刻模式的重复。如果某只股票倾向于在特定半小时槽位上涨，那么类似行为可能会在之后多个交易日的同一槽位再次出现。RTTA 将这种模式保存在 `slot_echo_[slot]` 中，即每个日内槽位残差收益的指数加权平均。

指标不会假定原始收益就是完整信号，而是先减去可选的 `market_return`：

```text
bar_return = log(close_t / close_{t-1})
residual_return = bar_return - market_return
```

如果提供市场或 ETF 收益，这样学习到的模式会更接近交易品种自身的相同时刻行为。

## 训练与在线状态

指标为每个槽位保存：

- `slot_echo`：该槽位残差收益的 EWMA；
- `slot_abs_err`：该槽位绝对残差收益的 EWMA；
- `slot_volume`：该槽位美元成交量的 EWMA；
- `slot_count`：该槽位的观测次数。

EWMA 学习率为：

```text
alpha = 2 / (lookback_days + 1)
```

`train(days)` 只是通过 `update(...)` 重放过去的 K 线，并在相邻交易日之间重置日内状态。这样既会更新每个槽位寿命较长的状态，也会更新滚动预测误差的校准状态。

`reset_session=True` 只清除日内状态：

- 上一根 K 线的对数收盘价；
- 尚待验证的预测；
- 最近一次结果。

它不会清除已经学习的槽位回声、槽位成交量或残差校准状态。

## 预测

每次更新时，指标通过查看未来槽位来预测接下来的 `horizon_bars`：

```text
future_slot_j = (current_slot + j) % slots_per_session
```

每个未来槽位的计算方式为：

```text
reliability = min(1, slot_count[future_slot] / min_slot_samples)
weight = exp(-0.10 * (j - 1)) * reliability
```

时钟回声是未来槽位回声的加权平均：

```text
clock_echo = sum(weight_j * slot_echo[future_slot_j]) / sum(weight_j)
prediction = clock_echo * horizon_bars
```

乘以 `horizon_bars`，是为了将平均每槽位残差收益转换为整个预测期限的预期对数收益。

## 流量与成交量调整

完成时钟预测后，当前 K 线的流量可以放大或减弱预测：

```text
dollar_volume = close * max(volume, 0)
normal_dv = supplied normal_dollar_volume, or slot_volume[slot], or dollar_volume
signed_flow = sign(bar_return) * dollar_volume / normal_dv
flow_confirm = sign(prediction) * signed_flow
prediction *= 1 + 0.20 * tanh(flow_confirm)
```

如果当前美元成交量与该槽位的正常美元成交量差异极大，预测会被削弱：

```text
volume_sync = log(dollar_volume / slot_volume[slot])
if abs(volume_sync) > 4:
    prediction *= 0.5
```

这样可以避免在交易活跃度严重异常时，过度信任相同时刻模式。

## 误差区间

预测就绪后，指标会保存：

- 距离预测到期还剩多少根 K 线；
- 入场时的对数收盘价；
- 预测值。

每次后续更新都会将待验证预测的剩余期限减一。预测到期时：

```text
realized = current_log_close - entry_log_close
realized_error = abs(realized - prediction)
```

误差会被推入滚动分位数。积累足够的校准样本后，半径为：

```text
radius = max(rolling_error_quantile, cost_buffer)
```

校准尚未就绪时，后备半径取决于该槽位的绝对残差收益：

```text
radius = max(cost_buffer, slot_abs_err[slot] * horizon_bars)
```

## 就绪状态

只有同时满足以下条件，`ready` 才为真：

- 相关未来槽位已有足够历史数据，可以产生加权预测；
- 滚动残差校准已积累足够多到期的预测误差。

如果 `fillna=False`，尚未就绪时会有意将输出留空：

- `prediction = NaN`
- `radius = NaN`
- `score = NaN`
- `signal = 0`
- `target_fraction = 0`

设置 `fillna=True` 后，会更早输出后备值。这对实验很有用，但实盘交易系统通常应当关注 `ready`。

## 交易输出

分数为：

```text
score = prediction / (radius + cost_buffer)
```

信号为：

```text
signal = +1 if score > entry_z
signal = -1 if allow_short and score < -entry_z
signal =  0 otherwise
```

目标仓位比例设有上限：

```text
target_fraction = max_abs_target_fraction * clamp(score / 3, side bounds)
```

`max_trade_dollars = participation_cap * normal_dollar_volume` 为执行层提供流动性参考。

## 输出

- `slot`：当前日内槽位。
- `samples_for_slot`：该槽位的历史观测数量。
- `bar_return`：当前相邻收盘价的对数收益。
- `residual_return`：K 线收益减去可选的市场收益。
- `clock_echo`：相同时刻残差收益的加权预测。
- `flow_confirm`：与预测方向一致的当前带符号流量。
- `volume_sync`：当前美元成交量相对槽位正常成交量的大小。
- `prediction`：预测期限内的预期对数收益。
- `radius`：经验不确定性区间。
- `score`：预测值除以误差半径与成本缓冲之和。
- `signal`：`-1`、`0` 或 `+1`。
- `target_fraction`：受上限约束的建议敞口。
- `max_trade_dollars`：流动性上限。
- `realized_error`：最近一条已经到期的预测误差。
- `ready`：历史数据与校准是否充足。

## 预期用途

本指标适合用于交易时段槽位稳定的 K 线。由于特征本身就是日内时刻，缺失 K 线、半日交易和错误的时段重置都会影响结果。在 Massive/Polygon 示例中，每个交易品种先使用之前多个交易日的聚合 K 线训练，然后按对齐后的窗口起始时刻对实时交易日评分。

在能够取得市场收益时，通常应将其与指标一同提供。如果不进行市场调整，广泛的日内市场波动可能会被错误地学习成某个交易品种特有的时钟行为。
