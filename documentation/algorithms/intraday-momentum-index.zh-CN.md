# 日内动量指数（IntradayMomentumIndex）

## 摘要

`IntradayMomentumIndex` 是 RTTA 对日内动量指数（IMI）的流式实现：一种 RSI 风格的振荡器，把每根 K 线的开盘到收盘涨幅与跌幅在滚动窗口内汇总比较。

## 更新 API

```python
result = rtta.IntradayMomentumIndex(window=14, fillna=True).update(open, close)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `window`  | `14`    | 涨幅和跌幅的滚动求和窗口 |
| `fillna`  | `True`  | 若为 `False`，窗口填满前返回 NaN |

当两个滚动和均有定义时，`update(open, close)` 返回 \([0,100]\) 范围内的标量 IMI 值。

## 工作原理

经典 RSI 使用收盘到收盘变化，IMI 则按每根 K 线的**日内**方向分类：

- 若 \(close>open\)，完整的开盘到收盘涨幅计为收益。
- 若 \(close<open\)，完整的开盘到收盘跌幅计为损失。
- 开盘价与收盘价相同，则收益和损失贡献均为零。

对 \(n\) 根 K 线的收益与损失作滚动求和，形成类似 RSI 的比率。IMI 较高表示近期多数 K 线收盘高于开盘（日内买入压力）；较低则表示相反。它常用于日内 K 线，或以日线 OHLC 衡量交易时段情绪。

与 Wilder RSI 不同，本实现对收益/损失序列使用**简单滚动和**，而不是 Wilder 平滑。

## 递推公式

令 \(o_t,c_t\) 为开盘价和收盘价，\(n\) 为窗口长度。

\[
G_t =
\begin{cases}
c_t - o_t & c_t > o_t \\
0 & \text{其他情况}
\end{cases}
,\qquad
L_t =
\begin{cases}
o_t - c_t & c_t < o_t \\
0 & \text{其他情况}
\end{cases}
\]

\[
IMI_t = 100 \cdot \frac{\sum_{i=0}^{n-1} G_{t-i}}{\sum_{i=0}^{n-1} G_{t-i} + \sum_{i=0}^{n-1} L_{t-i}}
\]

当 `fillna=True` 时，部分窗口使用目前已累积的样本。当 `fillna=False` 时，在收益缓冲区填满之前返回 NaN。即使两个总和均为零，安全除法也会产生有定义的值。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class IntradayMomentumIndex` 中实现。
- 两个 `RollingBuffer` 通过 `rolling_sum_push` 维护运行总和 `gain_sum_` / `loss_sum_`。
- 输出为标量 `double`，不是结果结构体。

## 参考资料

- [Investopedia——Intraday Momentum Index（IMI）](https://www.investopedia.com/terms/i/intraday-momentum-index-imi.asp)
- [IMI 概述](https://www.tradingview.com/support/solutions/43000589111-intraday-momentum-index/)
