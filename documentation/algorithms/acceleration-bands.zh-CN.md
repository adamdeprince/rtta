# AccelerationBands

## 摘要

`AccelerationBands` 是 RTTA 对 Price Headley 加速带的流式实现：上轨和下轨
分别由按价格区间缩放后的最高价与最低价的简单移动平均构成，中轨则是收盘价的
简单移动平均。

## 更新 API

```python
result = rtta.AccelerationBands(window=20, factor=4.0, fillna=True).update(close, high, low)
```

| 参数 | 默认值 | 含义 |
|------|-------:|------|
| `window` | `20` | 上轨、下轨与中轨的 SMA 周期 |
| `factor` | `4.0` | 区间缩放因子 \(f\) |
| `fillna` | `True` | 若为 `False`，在取得 `window` 个样本之前返回 NaN |

`update(...)` 返回包含 `middle`、`upper` 和 `lower` 字段的结果。
`advance(close, high, low)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

每根 K 线的最高价和最低价，会按照该线价格区间相对于中间区间
\(h+l\) 的比例向外扩张或向内收缩：

\[
s_t = f \frac{h_t - l_t}{h_t + l_t}.
\]

上轨源序列为 \(h_t(1+s_t)\)，下轨源序列为 \(l_t(1-s_t)\)。宽幅 K 线
会将两个源序列推得更远（即更强的“加速”），窄幅 K 线则使其收拢。对这两个
源序列分别计算 SMA，便得到绘图所用的上下轨；中轨是收盘价的普通 SMA，作为
参照。

价格突破上轨或跌破下轨，通常被解读为动量达到极端；价格触及中轨则可作为
均值回归的背景信息。

## 递推公式

令 \(c_t,h_t,l_t\) 分别为收盘价、最高价和最低价，\(n\) 为周期，\(f\) 为
缩放因子。

\[
s_t = f \cdot \frac{h_t - l_t}{h_t + l_t}
\]

（当 \(h_t+l_t=0\) 时，C++ 实现使用安全除法。）

\[
u^{src}_t = h_t(1 + s_t),\qquad
\ell^{src}_t = l_t(1 - s_t)
\]

\[
U_t = \operatorname{SMA}_n(u^{src}_t),\qquad
L_t = \operatorname{SMA}_n(\ell^{src}_t),\qquad
M_t = \operatorname{SMA}_n(c_t)
\]

结果中：`middle` \(= M_t\)、`upper` \(= U_t\)、`lower` \(= L_t\)。

当 `fillna=False` 且已取得的样本少于 \(n\) 个时，三个字段均为 NaN。
上下轨内部的 SMA 始终使用 `fillna=True`，因此可以形成不完整周期的平均值；
外层的 `fillna` 开关负责控制最终返回的结构体。

## 实现说明

- 实现在 `src/rtta/indicator.cpp` 的 `class AccelerationBands` 中。
- 结果类型为 `AccelerationBandsResult`（`middle`、`upper`、`lower`）。
- 使用三个相互独立的 `SMA` 实例：`upper_sma_`、`lower_sma_`、
  `middle_sma_`。
- 批量辅助函数为 `batch_acceleration_bands`。

## 参考资料

- [TradingView：Acceleration Bands](https://www.tradingview.com/support/solutions/43000589125-acceleration-bands/)
- [Investopedia：Price Headley 加速带概述](https://www.investopedia.com/terms/a/acceleration-bands.asp)
