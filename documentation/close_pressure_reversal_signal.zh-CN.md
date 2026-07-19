# ClosePressureReversalSignal

`ClosePressureReversalSignal` 是一种用于尾盘的横截面反转信号。它围绕一个明确的实证现象设计：日内跌幅较大的股票在临近收盘时可能承受价格压力，而这股压力可能在最后几根 K 线中部分反转。

主要参考论文为：

- Baltussen、Da 与 Soebhag，"End-of-Day Reversal"，SSRN，2024/2025。论文记录了交易日最后 30 分钟内股票收益的横截面反转现象，尤其关注日内输家在正向价格压力下出现的反转。SSRN 页面：https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5039009

RTTA 的实现并非逐字逐式复现论文，而是将论文中的思路转化为可在聚合 K 线上运行的增量信号，并可通过 `ClosePressureReversalUniverse` 用于横截面选股。

## API

```python
rtta.ClosePressureReversalSignal(
    cutoff_after_bars=66,
    entry_start_after_bars=72,
    entry_end_after_bars=75,
    exit_after_bars=77,
    calibration_window=120,
    calibration_quantile=0.80,
    reversal_slope=0.025,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.04,
    participation_cap=0.02,
    allow_short_winners=False,
    fillna=False,
    max_loser_z=6.0,
    max_range_z=5.0,
    max_volume_shock=6.0,
)
```

`update(open, high, low, close, volume, vwap=nan, transactions=nan, previous_session_close=nan, normal_dollar_volume=nan, normal_transactions=nan, reset_session=False)` 每次接收一根日内聚合 K 线。

默认时间参数假定美股常规交易时段为 390 分钟，并使用 5 分钟 K 线：

- 第 66 根 K 线为截点，约为美国东部时间下午 3:00；
- 第 72 至 75 根 K 线为入场窗口，约为下午 3:30 至 3:50；
- 第 77 根 K 线为退出窗口，约为收盘时刻。

若 K 线周期发生变化，也应相应调整这些时间参数。

## 交易时段状态

每个交易时段开始时，指标会重置：

- K 线计数；
- 上一根 K 线的对数收盘价；
- 锚定对数收盘价；
- 时段 VWAP 的累计状态；
- 截点之前的收益率累计量；
- 已冻结的当日剩余时段收益；
- 尚待验证的入场窗口预测。

除非调用 `reset()`，以下寿命较长的校准状态会继续保留：

- 已实现误差的滚动分位数；
- 正常美元成交量 EWMA；
- 正常成交笔数 EWMA；
- 价格区间 EWMA。

## 核心思路

论文研究的是横截面上的日终反转。RTTA 将其转化为逐根 K 线更新的分数：

1. 衡量股票截至截点时当日上涨或下跌了多少。
2. 用截至截点的已实现日内波动率对该变动进行标准化。
3. 当成交量和成交笔数异常偏高时，提高分数。
4. 当股票低于时段 VWAP 时，提高做多分数。
5. 只在配置的尾盘入场窗口内发出入场信号。
6. 在配置的退出窗口内强制退出。

默认实现偏向做多，因为论文重点讨论日内输家的反转。设置 `allow_short_winners=True` 后，会对日内赢家启用力度较弱的做空逻辑。

## 分步算法

对每一根有效 K 线：

1. 将 `bar_number` 加一。
2. 如果提供了 `previous_session_close`，以它建立 `anchor_log_close`；否则使用第一根 K 线的收盘价。
3. 计算相邻收盘价的对数收益 `ret1`。
4. 在截点 K 线及其之前，累计收益率的均值和方差项。
5. 计算当日剩余时段收益：

   ```text
   rod_return = log(close) - anchor_log_close
   ```

6. 到达 `cutoff_after_bars` 时冻结该数值：

   ```text
   frozen_rod_return = rod_return at cutoff
   ```

7. 根据截至截点的收益估计日内波动率：

   ```text
   intraday_vol = sqrt(var(ret1_to_cutoff) * count)
   ```

8. 将冻结收益转为输家和赢家 z 分数：

   ```text
   loser_z = max(0, -frozen_rod_return) / intraday_vol
   winner_z = max(0,  frozen_rod_return) / intraday_vol
   ```

9. 计算活跃度冲击：

   ```text
   volume_shock = log(dollar_volume / normal_dollar_volume)
   transaction_shock = log(transactions / normal_transactions)
   ```

10. 计算时段 VWAP 缺口：

    ```text
    vwap_gap = close / session_vwap - 1
    ```

11. 用当前最高价与最低价之差相对区间 EWMA 的大小，计算区间 z 分数。

12. 构造压力乘数：

    ```text
    volume_mult = 1 + 0.20 * clamp(volume_shock, -2, 4)
    tx_mult     = 1 + 0.10 * clamp(transaction_shock, -2, 4)
    ```

13. 对做多反转压力，奖励处于 VWAP 下方的输家：

    ```text
    long_vwap_mult = 1 + 0.50 * clamp((-vwap_gap) / intraday_vol, 0, 3)
    long_pressure_score = loser_z * volume_mult * tx_mult * long_vwap_mult
    ```

14. 如果允许做空赢家，则对处于 VWAP 上方的赢家计算一个力度较弱的做空压力分数：

    ```text
    short_vwap_mult = 1 + 0.50 * clamp(vwap_gap / intraday_vol, 0, 3)
    short_pressure_score = winner_z * volume_mult * tx_mult * short_vwap_mult
    ```

15. 将压力转为预期反转幅度：

    ```text
    long_prediction =
        reversal_slope * loser_return * clamp(long_pressure_score / 2, 0, 2)

    short_prediction =
       -0.50 * reversal_slope * winner_return * clamp(short_pressure_score / 2, 0, 2)
    ```

16. 当价格变动显得过于极端时阻止入场：

    ```text
    news_guard =
        loser_z > max_loser_z
        or winner_z > max_loser_z
        or range_z > max_range_z
        or volume_shock > max_volume_shock
    ```

## 误差区间

指标在入场窗口内作出预测后会保存该预测。到达或超过 `exit_after_bars` 时，它会比较从预测 K 线到当前 K 线的已实现对数收益与预测值：

```text
realized_error = abs(realized_return - prediction)
```

该误差被推入滚动分位数。校准样本足够后，`radius` 为：

```text
radius = max(rolling_error_quantile, cost_buffer)
```

校准尚未就绪时，使用以下后备半径：

```text
radius = max(2 * cost_buffer, max(0.0005, 1.25 * intraday_vol))
```

与 RTTA 的其他 moonshot 信号一样，这是一种受共形推断启发的经验校准方法，而非带覆盖率保证的正式论文复现。

## 交易输出

分数为：

```text
score = prediction / (radius + cost_buffer)
```

只有同时满足以下条件，才允许入场：

- 当前 K 线位于 `entry_window` 内；
- `news_guard` 为假；
- 分数越过 `entry_z`；
- 对应交易方向已启用。

进入退出窗口后，信号会被强制归零：

```text
signal = 0 if exit_window or news_guard or not entry_window
signal = +1 if prediction > entry_z * (radius + cost_buffer)
signal = -1 if allow_short_winners and prediction < -entry_z * (radius + cost_buffer)
```

`target_fraction` 随分数缩放，并受 `max_abs_target_fraction` 限制。`max_trade_dollars = participation_cap * normal_dollar_volume`。

## ClosePressureReversalUniverse

`ClosePressureReversalUniverse` 为每个交易品种保存一个 `ClosePressureReversalSignal`，并提供横截面筛选器：

```python
selected, exits = universe.update(
    indices, open, high, low, close, volume, vwap, transactions, top_fraction
)
```

对每一组同一时点的 K 线，它会：

- 更新传入的全部交易品种；
- 记录处于退出窗口的所有品种；
- 对入场窗口内分数为正的候选品种进行排名；
- 选出排名最高的 `ceil(candidate_count * top_fraction)` 个品种。

这是 Massive/Polygon 示例优先采用的 API，因为论文所描述的效应是横截面的。

## 输出

- `bar_number`：当前时段内从 1 开始的 K 线编号。
- `rod_return`：当前价格相对上一时段收盘价或首根 K 线锚点的收益。
- `frozen_rod_return`：在截点冻结的收益。
- `loser_z` / `winner_z`：标准化后的冻结跌幅/涨幅。
- `range_z`：当前 K 线区间相对区间 EWMA 的大小。
- `volume_shock`：美元成交量的对数冲击。
- `transaction_shock`：成交笔数的对数冲击。
- `vwap_gap`：收盘价相对时段 VWAP 的位置。
- `pressure_score`：当前交易方向的压力分数。
- `prediction`：预期反转对数收益。
- `radius`：经验不确定性区间。
- `score`：预测值除以误差半径与成本缓冲之和。
- `signal`：`-1`、`0` 或 `+1`。
- `target_fraction`：受上限约束的建议敞口。
- `max_trade_dollars`：流动性上限。
- `realized_error`：最近一条已经到期的入场至退出预测误差。
- `entry_window`、`exit_window`、`frozen`、`news_guard`：布尔状态标志。

## 预期用途

本指标应使用间隔规则的日内 K 线，并在正确的交易时段边界重置。对美股而言，默认参数面向 5 分钟 K 线。示例脚本对交易品种进行横截面排名，并使用报价模拟延迟成交。指标本身不会下单，也不了解融券、停牌、涨跌停或投资组合约束。
