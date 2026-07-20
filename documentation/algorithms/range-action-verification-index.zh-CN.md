# 区间行情验证指数（RangeActionVerificationIndex）

## 摘要

`RangeActionVerificationIndex`（RAVI）是 RTTA 对 Tushar Chande 区间行情验证指数的流式实现：短周期与长周期简单移动平均线之间的绝对距离，以长期 SMA 的百分比表示。

## 更新 API

```python
value = rtta.RangeActionVerificationIndex(
    short_window=7, long_window=65, fillna=True
).update(close)
```

每次调用 `update(...)` 处理一个收盘价。`advance(...)` 更新状态但不返回 Python 值。标量 `batch(...)` 返回一个 NumPy 数组。

## 工作原理

RAVI 与 ADX 同属判断趋势是否存在的指标：快速与慢速均线之间距离较大，表示市场具有明确方向；距离较小则表示区间震荡。与 ADX 不同，RAVI 不使用真实波幅或方向运动——只使用 SMA 的相对距离。

## 递推公式

令 \(x_t\) 为收盘价，\(n_s\) 为 `short_window`，\(n_\ell\) 为 `long_window`。

\[
S_t = \operatorname{SMA}_{n_s}(x_t), \qquad
L_t = \operatorname{SMA}_{n_\ell}(x_t)
\]

\[
\operatorname{RAVI}_t = 100\cdot\frac{\lvert S_t - L_t\rvert}{L_t}
\]

当 `fillna=False` 时，在长期窗口填满之前输出为 `NaN`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class RangeActionVerificationIndex` 中实现，包含两个内部 `SMA` 成员。

## 参考资料

- [TradingPedia：Chande's Range Action Verification Index（RAVI）](https://www.tradingpedia.com/forex-trading-indicators/chandes-range-action-verification-index/)
- [Wealth-Lab Wiki：RAVI](http://www2.wealth-lab.com/wl5WIKI/RAVI.ashx)
