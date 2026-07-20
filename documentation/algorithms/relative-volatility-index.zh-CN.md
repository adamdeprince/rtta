# 相对波动率指数（RelativeVolatilityIndex）

## 摘要

`RelativeVolatilityIndex` 是 RTTA 对 Donald Dorsey 相对波动率指数（RVI）的流式实现。它以 RSI 风格的方式平滑收盘价上行和下行标准差，并输出 RVI 的 EMA 信号线。

## 更新 API

```python
result = rtta.RelativeVolatilityIndex(
    std_window=10, smooth_window=14, fillna=True
).update(close)
# result.rvi, result.signal
```

当 `fillna=False` 时，在取得 `std_window + smooth_window` 个样本之前，两个字段均为 `NaN`。

## 工作原理

Dorsey 的 RVI 外形类似 RSI，但把价格变化替换为收盘价滚动标准差：收盘上涨时归入上行侧，收盘下跌时归入下行侧。因此，它衡量波动率的方向，而不是价格的方向：RVI 较高，表示波动更多发生在上涨过程中；较低则表示更多发生在下跌过程中。RTTA 还用相同长度的 EMA 平滑 RVI，作为信号线。

## 递推公式

令 \(c_t\) 为收盘价，\(n_s\) 为 `std_window`（默认 \(10\)），\(n_e\) 为 `smooth_window`（默认 \(14\)）。

\[
\sigma_t = \operatorname{StdDev}_{n_s}(c_t)
\]

把波动率分配到上行/下行两侧（第一根 K 线贡献为零）：

\[
(u_t, d_t) =
\begin{cases}
(\sigma_t,\, 0), & c_t > c_{t-1} \\
(0,\, \sigma_t), & c_t < c_{t-1} \\
(0.5\,\sigma_t,\, 0.5\,\sigma_t), & c_t = c_{t-1}
\end{cases}
\]

\[
U_t = \operatorname{EMA}_{n_e}(u_t), \qquad
D_t = \operatorname{EMA}_{n_e}(d_t)
\]

\[
RVI_t = 100 \cdot \frac{U_t}{U_t + D_t}
\quad\text{（安全除法）}
\]

\[
\operatorname{signal}_t = \operatorname{EMA}_{n_e}(RVI_t)
\]

内部 StdDev 和 EMA 使用 `fillna=True`。外层预热计数为 \(n_s+n_e\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class RelativeVolatilityIndex` 中实现，成员包括 `std_`、`up_`、`down_` 和 `signal_`。结果字段为 `rvi` 和 `signal`。另请参阅对 RVI 作回归的 [`Inertia`](inertia.zh-CN.md)。

## 参考资料

- [Investopedia：Relative Volatility Index](https://www.investopedia.com/terms/r/relative-volatility-index-rvi.asp)
- [TradingView：Relative Volatility Index](https://www.tradingview.com/scripts/relativevolatilityindex/)
