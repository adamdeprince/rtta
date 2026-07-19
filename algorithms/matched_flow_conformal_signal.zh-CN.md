# MatchedFlowConformalSignal

**状态：** 拟议中的 RTTA 指标 / 研究原型  
**目标模块：** `rtta.indicator`  
**建议的 C++ 类：** `MatchedFlowConformalSignal`  
**建议的结果结构体：** `MatchedFlowConformalSignalResult`

`MatchedFlowConformalSignal` 是一种面向 5 分钟 OHLCV K 线的中速技术交易信号。它将近期价格变动、价格相对 VWAP 的位置、带符号成交量流、相对成交量和在线误差区间组合成简单的交易输出：

```text
prediction        对未来一个预测期限的对数收益估计
radius            根据近期误差/噪声自适应的区间
score             预测值 / 不确定性
signal            -1、0 或 +1
target_fraction   建议的投资组合比例
max_trade_dollars 执行层的流动性上限
```

本设计面向已经接好市场数据和下单 API 的开发者，例如使用 Polygon/Massive 获取 K 线、使用 Alpaca 执行交易。RTTA 对象本身**不应**下单；它只应输出有状态的技术分析结果，供执行层使用。

这**不是**财务建议，也不是保证盈利的策略。它只是将若干当前研究思路清晰、可测试地转化为增量式 C++ 指标。

---

## 一句话概括

只有当近期方向性流量、动量和 VWAP 位置共同表明未来一小时可能延续，**而且**预测变动大于指标自身近期预测误差时，才进行交易。

用程序员熟悉的方式表达：

```text
if prediction > recent_error_band + estimated_cost:
    emit LONG
elif prediction < -(recent_error_band + estimated_cost):
    emit SHORT
else:
    emit FLAT
```

真正有用的是 `recent_error_band`。它可以阻止指标对每一次微小的噪声波动都进行交易。

---

## 为什么它适合 RTTA

RTTA 的定位是超低延迟增量技术分析工具包。README 说明，本软件包不采用只能批处理的 pandas 式计算，而是每次用一个样本更新指标。新的多输出指标应公开 `update(...)`、`advance(...)`、不可变结果结构体、标量 `update_<field>(...)` 方法和 `last_<field>()` 访问器。

`MatchedFlowConformalSignal` 符合这一模式，因为它是有状态且增量式的：

1. 每次调用恰好接收一根 OHLCV K 线。
2. 它以 O(1) 或接近 O(1) 的时间复杂度更新滚动状态。
3. 它输出含多个只读字段的结果结构体。
4. 它在内部保留旧预测，之后再与已实现收益进行比较。
5. 运行时不需要 pandas、Python 回调或完整的机器学习模型。

---

## 输入

建议的 `update(...)` 签名：

```cpp
MatchedFlowConformalSignalResult update(
    double open,
    double high,
    double low,
    double close,
    double volume,
    double normal_dollar_volume = NaN,
    double market_cap = NaN,
    bool reset_session = false
);
```

### 必需输入

| 输入 | 含义 |
|---|---|
| `open` | K 线开盘价。目前主要为了保持 API 对称，并未大量使用。 |
| `high` | K 线最高价，用于计算价差/噪声代理。 |
| `low` | K 线最低价，用于计算价差/噪声代理。 |
| `close` | K 线收盘价，也是主要的价格输入。 |
| `volume` | K 线成交股数，用于计算美元成交量和流量。 |

### 可选输入

| 输入 | 含义 |
|---|---|
| `normal_dollar_volume` | 相同时刻的历史美元成交量中位数或平均数。例如，某交易品种过去 20 个交易日 10:35 的美元成交量中位数。用于计算相对成交量和执行上限。 |
| `market_cap` | 当前市值。若提供，可按市值对带符号流量进行归一化；若缺失，指标会改用正常美元成交量归一化。 |
| `reset_session` | 在常规交易时段第一根 K 线上设为 `true`，使 VWAP、日内滞后值和待验证的日内预测得到干净重置。 |

---

## 输出

建议的结果结构体：

```cpp
struct MatchedFlowConformalSignalResult {
    double prediction;
    double radius;
    double score;
    double signal;
    double target_fraction;

    double alpha_flow;
    double participation;
    double flow_score;
    double momentum;
    double volatility;
    double vwap_gap;
    double rel_dollar_volume;

    double max_trade_dollars;
    double realized_error;
};
```

### 主要字段

| 字段 | 含义 |
|---|---|
| `prediction` | 配置期限内的对数收益估计；通常 12 根 5 分钟 K 线，即 60 分钟。 |
| `radius` | 近期预测误差的分位数，即噪声/不确定性区间。 |
| `score` | `prediction / (radius + cost_buffer)`。绝对值越大，确信程度越高。 |
| `signal` | `+1` 做多、`0` 空仓、`-1` 做空。 |
| `target_fraction` | 建议的投资组合比例。执行代码可以将它乘以账户权益。 |
| `max_trade_dollars` | 根据正常美元成交量和 `participation_cap` 计算的每根 K 线名义金额上限。 |

### 诊断字段

| 字段 | 含义 |
|---|---|
| `alpha_flow` | 为提取 alpha 而归一化的带符号美元流量；有市值时使用市值。 |
| `participation` | 按正常美元成交量归一化的带符号美元流量，可用于分析执行与流动性。 |
| `flow_score` | 经压缩的滚动流量分数。 |
| `momentum` | 近期对数收益动量的混合值。 |
| `volatility` | 近期收益波动率。 |
| `vwap_gap` | `(close / session_vwap) - 1`。正值表示价格高于时段 VWAP。 |
| `rel_dollar_volume` | 当前美元成交量除以相同时刻的正常美元成交量。 |
| `realized_error` | 旧预测到期后得到的误差（如有）。 |

---

## 算法详解

假设使用 5 分钟 K 线，且 `horizon_bars = 12`，那么信号要估计的是未来 60 分钟。

### 1. 在每个交易时段开始时重置日内状态

在常规交易时段第一根 K 线上调用：

```cpp
sig.update(open, high, low, close, volume, normal_dollar_volume, market_cap, true);
```

这会重置时段 VWAP、收盘价滞后值、短期滚动统计量和待验证预测。除非用户调用 `reset()`，否则**不应**清除寿命较长的校准/误差窗口。

原因在于，该指标面向日内行为。如果让前一交易日的收盘价或 VWAP 泄漏到今天最初几根 K 线中，信号会更难解释。

---

### 2. 计算基础 K 线数值

```text
dollar_volume = close * volume
ret_1         = log(close / previous_close)
signed_bar    = sign(ret_1)
```

`signed_bar` 是一种简单的交易方向代理。如果当前 K 线相对上一根收涨，美元成交量被视为正向流量；如果收跌，则视为负向流量。

它不如报价层面的订单流失衡精确，但普通 OHLCV K 线即可计算。

---

### 3. 维护时段 VWAP

```text
session_vwap = cumulative(close * volume) / cumulative(volume)
vwap_gap     = close / session_vwap - 1
```

解释如下：

```text
vwap_gap > 0  -> 价格高于当日平均成交价
vwap_gap < 0  -> 价格低于当日平均成交价
```

这为信号提供了一个简单的位置特征。价格高于 VWAP 时出现的强劲上行流量，与价格仍低于 VWAP 时的小幅反弹，应得到不同处理。

---

### 4. 计算两种带符号流量

指标保留两种相互关联但有意区分的流量数值。

#### Alpha 流量

```text
alpha_flow = sign(ret_1) * dollar_volume / market_cap
```

如果缺少 `market_cap`：

```text
alpha_flow = sign(ret_1) * dollar_volume / normal_dollar_volume
```

该数值用于提取方向性信号。

按市值归一化来自匹配滤波订单流的思路：知情流量可能随公司价值变化，而原始日成交量可能引入换手率噪声。这一观点并未针对所有美股日内应用得到验证，因此应通过消融实验检验该特征。

#### 参与流量

```text
participation = sign(ret_1) * dollar_volume / normal_dollar_volume
```

该数值用于分析执行与流动性。

参与率式数值回答的是：“与一天中这个时刻的正常活跃度相比，这根带符号 K 线有多大？”

---

### 5. 计算短期动量与波动率

对 5 分钟 K 线，建议使用以下滚动窗口：

```text
mom_3  = log(close / close_3_bars_ago)    # 15 分钟
mom_6  = log(close / close_6_bars_ago)    # 30 分钟
mom_12 = log(close / close_12_bars_ago)   # 60 分钟

vol_12 = stddev(last 12 one-bar log returns)
```

然后混合动量：

```text
momentum = 0.20 * mom_3 + 0.35 * mom_6 + 0.45 * mom_12
```

这些权重有意保持简单，既便于调优，也便于解释。

---

### 6. 将滚动流量转换为有界分数

例如：

```text
flow_score = tanh(alpha_flow_12_sum / alpha_norm
                  + 0.50 * participation_6_sum / part_norm)
```

为什么使用 `tanh`？因为它能防止一根极端成交量 K 线永久支配整个指标。

---

### 7. 估计未来一个预测期限的收益

采用一套有意保持简单的启发式预测：

```text
activity_score = tanh((rel_dollar_volume - 1.0) / 2.0)
spread_proxy   = max(0, high - low) / close
spread_dampen  = 1 / (1 + 25 * spread_proxy)

raw_prediction = 0.35   * momentum
               + 0.0010 * flow_score
               + 0.05   * vwap_gap
               + 0.0005 * activity_score

prediction = spread_dampen * raw_prediction
```

这是对训练模型的技术分析近似，有意保持确定性和高速。

预测值使用对数收益单位。例如：

```text
prediction = 0.0020
```

大致表示：

```text
未来一小时的估计收益为 +0.20%
```

---

### 8. 保存预测，之后再测量误差

每当指标输出预测时，保存：

```text
pending_prediction = { close_now, prediction_now }
```

经过 `horizon_bars` 次更新后，将其与已实现收益进行比较：

```text
realized_return = log(close_now / old_prediction.close)
error           = abs(realized_return - old_prediction.prediction)
```

将 `error` 推入滚动校准窗口。

关键就在这里：指标会不断追问“我最近错得有多厉害？”

---

### 9. 构造类共形误差区间

保留最近 `calibration_window` 个预测绝对误差，然后计算较高分位数，例如 80%：

```text
radius = quantile(abs_errors, 0.80)
```

该 `radius` 就是不确定性区间。

例如：

```text
prediction = +0.0030
radius     =  0.0018
cost       =  0.0005
```

预测变动大于近期模型误差与估计成本之和，因此允许做多。

---

### 10. 生成信号

```text
denom = radius + cost_buffer
score = prediction / denom

if prediction > entry_z * denom:
    signal = +1
elif prediction < -entry_z * denom:
    signal = -1
else:
    signal = 0
```

默认值：

```text
entry_z = 1.0
cost_buffer = 0.0005
```

在对数收益单位中，`cost_buffer = 0.0005` 约等于 5 个基点。实际使用时，应改为自己对价差、费用、滑点和跨价成本的估计。

---

### 11. 建议目标仓位

```text
if signal != 0:
    target_fraction = max_abs_target_fraction * clamp(score / 3, -1, +1)
else:
    target_fraction = 0
```

默认值：

```text
max_abs_target_fraction = 0.05
```

因此，默认情况下指标对单个交易品种的建议账户配置比例永远不会超过 5%。

---

### 12. 限制每根 K 线的执行规模

```text
max_trade_dollars = participation_cap * normal_dollar_volume
```

默认值：

```text
participation_cap = 0.02
```

如果当前 5 分钟槽位的正常美元成交量为 `$2,000,000`，执行层在这根 K 线上交易的金额不应超过：

```text
0.02 * 2,000,000 = $40,000
```

指标**不了解**账户当前持仓。执行层应自行计算差额：

```text
target_notional = account_equity * result.target_fraction
current_notional = current_shares * last_price
wanted_delta = target_notional - current_notional

order_notional = clamp(
    wanted_delta,
    -result.max_trade_dollars,
    +result.max_trade_dollars
)
```

---

## 默认参数

| 参数 | 建议默认值 | 含义 |
|---|---:|---|
| `horizon_bars` | `12` | 预测期限。对 5 分钟 K 线而言即 60 分钟。 |
| `calibration_window` | `250` | 用于计算滚动误差区间的到期误差数量。 |
| `calibration_quantile` | `0.80` | 用作 `radius` 的误差分位数。数值越高，交易越少。 |
| `entry_z` | `1.0` | 相对 `radius + cost` 所需的预测强度。 |
| `cost_buffer` | `0.0005` | 以收益率表示的往返交易摩擦估计。 |
| `max_abs_target_fraction` | `0.05` | 单个交易品种的最大建议配置比例。 |
| `participation_cap` | `0.02` | 每根 K 线最大交易名义金额占正常美元成交量的比例。 |
| `fillna` | `false` | 遵循 RTTA 约定：状态未填充时返回 NaN，除非 `fillna=true`。 |

---

## 预期的 RTTA API

### C++ 类

```cpp
class MatchedFlowConformalSignal {
public:
    MatchedFlowConformalSignal(
        std::size_t horizon_bars = 12,
        std::size_t calibration_window = 250,
        double calibration_quantile = 0.80,
        double entry_z = 1.0,
        double cost_buffer = 0.0005,
        double max_abs_target_fraction = 0.05,
        double participation_cap = 0.02,
        bool fillna = false
    );

    MatchedFlowConformalSignalResult update(
        double open,
        double high,
        double low,
        double close,
        double volume,
        double normal_dollar_volume = NaN,
        double market_cap = NaN,
        bool reset_session = false
    );

    void advance(...same args...);

    double update_prediction(...same args...);
    double update_radius(...same args...);
    double update_score(...same args...);
    double update_signal(...same args...);
    double update_target_fraction(...same args...);

    MatchedFlowConformalSignalResult last() const;
    double last_prediction() const;
    double last_radius() const;
    double last_score() const;
    double last_signal() const;
    double last_target_fraction() const;
    double last_max_trade_dollars() const;
    double last_realized_error() const;

    void reset();
    void reset_intraday();
};
```

### Python 用法

```python
from rtta.indicator import MatchedFlowConformalSignal

sig = MatchedFlowConformalSignal(
    horizon_bars=12,
    calibration_window=250,
    calibration_quantile=0.80,
    entry_z=1.0,
    cost_buffer=0.0005,
    max_abs_target_fraction=0.05,
    participation_cap=0.02,
    fillna=True,
)

for bar in bars:
    result = sig.update(
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume,
        normal_dollar_volume=bar.normal_dollar_volume,
        market_cap=bar.market_cap,
        reset_session=bar.is_first_regular_session_bar,
    )

    if result.signal > 0:
        print("LONG", result.score, result.target_fraction, result.max_trade_dollars)
    elif result.signal < 0:
        print("SHORT", result.score, result.target_fraction, result.max_trade_dollars)
    else:
        print("FLAT", result.score)
```

---

## 执行层示意

指标不应导入 Alpaca、Polygon/Massive、pandas 或任何券商客户端，应保持纯粹。

券商执行层可以这样使用输出：

```python
target_notional = account_equity * result.target_fraction
current_notional = current_qty * latest_price

delta = target_notional - current_notional

delta = max(
    -result.max_trade_dollars,
    min(result.max_trade_dollars, delta),
)

shares = int(delta / latest_price)

if shares > 0:
    submit_buy(symbol, shares)
elif shares < 0:
    submit_sell(symbol, abs(shares))
```

以下实际执行规则应放在指标之外：

1. 先使用模拟交易。
2. 开盘后最初几根 K 线应禁止交易，等待状态预热完成。
3. 如果策略仅用于日内交易，应在收盘前平仓或减仓。
4. 强制执行账户级敞口限制。
5. 强制执行交易品种级别的融券/做空规则。
6. 处理券商错误、部分成交、拒单和停牌。
7. 记录每个信号和下单决定。

---

## 测试内容

### 单元测试

1. **NaN 行为**  
   当 `fillna=false` 时，在对象状态填充完成前，`prediction`、`radius` 和 `score` 的早期输出应为 NaN。

2. **交易时段重置**  
   调用 `reset_session=true` 应重置 VWAP、收盘价滞后值、待验证预测和短期滚动窗口。

3. **校准到期**  
   经过 `horizon_bars` 次更新后，旧预测应到期，并将绝对误差推入残差窗口。

4. **信号阈值**  
   如果 `prediction <= radius + cost_buffer`，信号应为空仓；若超过阈值，则应具有方向。

5. **参与率上限**  
   提供正常美元成交量时，`max_trade_dollars` 应等于 `participation_cap * normal_dollar_volume`。

6. **缺失市值时的后备逻辑**  
   如果 `market_cap` 为 NaN，`alpha_flow` 应改用正常美元成交量归一化。

### 回测诊断

至少记录：

```text
symbol
timestamp
close
prediction
radius
score
signal
target_fraction
max_trade_dollars
realized_error
future_return
```

然后报告：

```text
交易次数
平均持有期
毛损益
扣除估计成本后的净损益
胜率
平均盈利
平均亏损
最大回撤
按交易品种拆分的损益
按时刻拆分的损益
按时刻拆分的信号数量
校准覆盖率：percent(|future_return - prediction| <= radius)
```

校准覆盖率检查尤其重要。如果 `calibration_quantile=0.80`，那么在稳定评估期内，到期绝对误差大约应有 80% 落在半径之内。由于漂移等原因，它不会精确等于 80%，但偏差很大就是警示信号。

---

## 已知局限

1. **C++ 对象并不等同于 Python 机器学习模型。**  
   原始思路使用 scikit 风格训练。本 RTTA 版本是受该流程启发的确定性流式指标。

2. **带符号 OHLCV 流量只是一种代理。**  
   `sign(ret_1) * dollar_volume` 并不是真正的订单流失衡。真正的 OFI 需要买卖报价更新及其挂单量。

3. **市值归一化在日内场景中仍属实验性。**  
   匹配滤波论文测试的是韩国股票的日度观测。将该思路用于美股 5 分钟 K 线是一项研究假设，而非已经证明的事实。

4. **误差区间是类共形方法，不是正式保证。**  
   滚动残差分位数借鉴了共形校准思路，但简化指标没有实现引用论文中的完整理论。

5. **成本可能占据主导。**  
   如果价差/滑点大于 `cost_buffer`，信号可能在回测中表现良好，却在实盘中失败。

6. **状态突变仍会造成伤害。**  
   滚动误差区间会自适应，但突发新闻、停牌、财报、美联储事件或全市场冲击仍可能使其失效。

---

## 建议的消融实验

消融实验是判断指标是否真正有效的最快方式。

1. **不使用共形过滤**  
   交易所有非零预测，并与过滤后的版本比较。

2. **不使用市值归一化**  
   仅以参与率式流量取代 `alpha_flow`。

3. **仅使用动量**  
   移除流量、VWAP 缺口和活跃度分数。

4. **仅使用流量**  
   移除动量和 VWAP 缺口。

5. **不同预测期限**  
   测试 `horizon_bars = 6`、`9`、`12` 和 `18`。

6. **不同分位数**  
   测试 `calibration_quantile = 0.70`、`0.80`、`0.90`。

7. **日内时段过滤**  
   比较全天交易与跳过开盘及收盘前一小时的结果。

---

## 文档与研究参考资料

### RTTA 实现参考

| 参考资料 | 相关原因 |
|---|---|
| [RTTA 分支 `0.2.0`](https://github.com/adamdeprince/rtta/tree/0.2.0) | 目标仓库与分支。 |
| [RTTA README](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/README.md) | 说明 RTTA 的增量更新目标、C++23/nanobind 构建方式、`fillna` 约定和指标 API 约定。 |
| [RTTA `ALGOS.md`](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/ALGOS.md) | 应加入本算法的公共指标清单。 |
| [RTTA `indicator.cpp`](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/src/rtta/indicator.cpp) | 现有 C++/nanobind 指标实现文件。 |
| [RTTA `__init__.py`](https://raw.githubusercontent.com/adamdeprince/rtta/0.2.0/src/rtta/__init__.py) | 可能需要公开新绑定的 Python 导出文件。 |
| [nanobind 类文档](https://nanobind.readthedocs.io/en/latest/classes.html) | 将 C++ 类公开给 Python 时所用的绑定风格。 |

### 市场数据与执行参考

| 参考资料 | 相关原因 |
|---|---|
| [Massive/Polygon Python 客户端](https://github.com/massive-com/client-python) | Polygon.io/Massive 数据的官方 Python 客户端，可用于外部数据管线获取 OHLCV K 线。 |
| [Massive/Polygon 聚合 K 线文档](https://github.com/polygon-io/client-python/blob/master/docs/source/Aggs.rst) | 说明 `RESTClient.list_aggs`，可用于生成 5 分钟 K 线。 |
| [Alpaca-py 下单 API](https://alpaca.markets/sdks/python/api_reference/trading/orders.html) | 说明如何在独立执行层中调用 `TradingClient.submit_order(...)`。 |
| [Alpaca 订单使用指南](https://docs.alpaca.markets/docs/working-with-orders) | 实际订单管理文档。 |

### 原始及补充研究

| 参考资料 | 本设计采用的思路 |
|---|---|
| [The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting](https://arxiv.org/abs/2602.03395) | 支持测试中间目标期限，而不是默认最终交易期限就是最佳训练标签。它对 Python 机器学习版本更重要，但也说明 C++ 默认期限为何应可调。 |
| [Optimal Signal Extraction from Order Flow: A Matched Filter Perspective on Normalization and Market Microstructure](https://arxiv.org/abs/2512.18648) | 支持将用于信号提取的市值归一化流量，与用于执行/流动性分析的成交量归一化参与率分开。 |
| [Temporal Conformal Prediction: A Distribution-Free Statistical and Machine Learning Framework for Adaptive Risk Forecasting](https://arxiv.org/abs/2507.05470) | 支持在非平稳金融时间序列中对预测不确定性进行滚动/自适应校准。 |
| [Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR](https://arxiv.org/abs/2602.03903) | 支持按近期程度和状态加权的共形校准。当前指标采用更简单的滚动分位数版本。 |
| [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170) | 为分布变化下的在线类共形自适应提供通用基础。 |
| [Online Conformal Model Selection for Nonstationary Time Series](https://arxiv.org/abs/2506.05544) | 支持静态模型选择在非平稳环境中可能很脆弱这一更广泛的观点，与未来的多专家版本更相关。 |
| [Forecasting Intraday Volume in Equity Markets with Machine Learning](https://arxiv.org/abs/2505.08180) | 支持使用预期日内成交量来设定执行上限，并在未来以学习得到的成交量预测替代 `normal_dollar_volume`。 |
| [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | 经典订单流失衡参考；如果将本 OHLCV 指标升级为报价层面的 OFI，会很有用。 |
| [Order-Flow Filtration and Directional Association with Short-Horizon Returns](https://arxiv.org/abs/2507.22712) | 支持在按方向使用高频订单流信号前先滤除噪声，可用于报价层面的扩展。 |

---

## 未来扩展

### 1. 报价层面的 OFI 版本

将 OHLCV 带符号流量：

```text
sign(ret_1) * dollar_volume
```

替换为根据最优买卖报价和挂单量更新计算的真正订单流失衡。

这需要不同的 update 签名，例如：

```cpp
update_quote(
    double bid_price,
    double bid_size,
    double ask_price,
    double ask_size,
    double trade_price,
    double trade_size,
    ...
)
```

### 2. 加权共形半径

用指数加权残差替换简单滚动分位数：

```text
weight_i = exp(-age_i / tau)
radius   = weighted_quantile(errors, weights, calibration_quantile)
```

这样可以在状态突变后更快适应。

### 3. 状态感知半径

加入一个简单状态特征：

```text
regime = recent_volatility / long_volatility
```

然后同时按时间远近和状态相似度为旧残差加权：

```text
weight_i = exp(-age_i / tau) * exp(-gamma * abs(regime_i - regime_now))
```

这更接近按状态加权的共形风险控制研究。

### 4. 离线学习系数

当前系数是固定的：

```text
0.35, 0.0010, 0.05, 0.0005
```

未来版本可以将它们公开为构造参数，或在 Python 中离线拟合后再传给 C++ 指标。

### 5. 成交量预测输入

不再传入相同时刻的美元成交量中位数，而是传入独立模型的预测：

```text
normal_dollar_volume = predicted_next_bar_or_next_hour_dollar_volume
```

这样 `max_trade_dollars` 可以对异常交易日作出更灵敏的反应。

---

## 建议加入 `ALGOS.md` 的条目

```markdown
| `MatchedFlowConformalSignal` | 中速 OHLCV 信号，结合日内动量、VWAP 缺口、带符号流量、相对成交量和在线残差/误差区间，输出预测、分数、方向、目标仓位比例和执行上限。 | https://arxiv.org/abs/2507.05470 |
```

---

## 建议的 Codex 任务

```text
新增一个名为 MatchedFlowConformalSignal 的 RTTA 指标。

采用 docs/MatchedFlowConformalSignal.md 中的设计。

要求：
1. 在 src/rtta/indicator.cpp 中加入 MatchedFlowConformalSignalResult 和 MatchedFlowConformalSignal。
2. 遵循 RTTA 现有的多输出指标约定。
3. 公开 update(...)、advance(...)、update_prediction(...)、update_radius(...)、update_score(...)、update_signal(...)、update_target_fraction(...)。
4. 公开 last()、last_prediction()、last_radius()、last_score()、last_signal()、last_target_fraction()、last_max_trade_dollars()、last_realized_error()。
5. 使用 nanobind 加入 Python 绑定。
6. 如果当前包结构需要，在 src/rtta/__init__.py 中导出这些名称。
7. 在 ALGOS.md 中加入对应条目。
8. 添加单元测试，覆盖预热/fillna 行为、交易时段重置、残差到期、信号阈值和参与率上限。
9. 不要在指标中加入券商/执行代码。
10. 不要在 C++ 热路径中加入 pandas 或 Python 回调。
```
