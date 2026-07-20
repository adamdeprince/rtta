# WaveTrend 振荡器

## 摘要

`WaveTrend` 是 RTTA 对 LazyBear WaveTrend 振荡器的流式实现：对 HLC3 的通道指数作双重平滑，产生 `wt1` / `wt2` 两条线，用于超买/超卖和交叉信号。

## 更新 API

```python
result = rtta.WaveTrend(
    channel_length=10, average_length=21, signal_length=4, fillna=True,
).update(high, low, close)
```

| 参数 | 默认值 | 含义 |
|-------------------|---------|---------|
| `channel_length`  | `10`    | ESA 和绝对偏差的 EMA 长度 |
| `average_length`  | `21`    | WT1 的 EMA 长度 |
| `signal_length`   | `4`     | WT2 信号线的 SMA 长度 |
| `fillna`          | `True`  | 若为 `False`，组合预热结束前返回 NaN |

`update(high, low, close)` 返回 `wt1`、`wt2`。`advance(...)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

WaveTrend 与作用于典型价格的商品通道指数密切相关，但增加了指数平滑：

1. 典型价格 \(ap=(h+l+c)/3\)。
2. ESA：\(ap\) 的 EMA。
3. \(ap\) 相对 ESA 的平均绝对偏差，同样用 EMA 平滑。
4. 通道指数 \(ci=(ap-esa)/(0.015\cdot d)\)。
5. WT1：\(ci\) 的 EMA；WT2：WT1 的 SMA（默认为 4）。

WT1 穿越 WT2 的交叉，以及极值区（LazyBear 图表常用 ±60 / ±53 附近），可用于均值回归和动量择时。常数 `0.015` 与 LazyBear / TradingView 参考实现的缩放一致。

## 递推公式

\[
ap_t = \frac{h_t + l_t + c_t}{3}
\]

\[
esa_t = \operatorname{EMA}_{n_1}(ap_t)
\]

\[
d_t = \operatorname{EMA}_{n_1}\big(|ap_t - esa_t|\big)
\]

\[
ci_t = \frac{ap_t - esa_t}{0.015\, d_t}
\]

（当 \(d_t=0\) 时使用安全除法。）

\[
wt1_t = \operatorname{EMA}_{n_2}(ci_t)
\]

\[
wt2_t = \operatorname{SMA}_{n_s}(wt1_t)
\]

默认值：\(n_1=10\)、\(n_2=21\)、\(n_s=4\)。

当 `fillna=False` 时，取得 \(n_1+n_2+n_s\) 个样本后才返回非 NaN 结果（保守预热，与 C++ 的 `warm_` 总和一致）。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class WaveTrend` 中实现。
- 内部组件：`EMA esa_`、`EMA d_`、`EMA wt1_`、`SMA wt2_`（嵌套平滑器均使用 `fillna=True`）。
- 结果类型：`WaveTrendResult`（`wt1`、`wt2`）。
- 批量辅助函数：`batch_wave_trend`。

## 参考资料

- [TradingView——LazyBear WaveTrend Oscillator（WT）](https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/)
- [WaveTrend 社区文档](https://www.tradingview.com/scripts/wavetrend/)
