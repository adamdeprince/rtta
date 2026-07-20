# 趋势强度指数（TrendIntensityIndex）

## 摘要

`TrendIntensityIndex` 是 RTTA 对趋势强度指数（TII）的流式实现：滚动窗口内为正的收盘价—SMA 偏差，占绝对偏差总量的百分比，缩放到 \(0\)–\(100\)。

## 更新 API

```python
value = rtta.TrendIntensityIndex(window=30, fillna=True).update(close)
```

当 `fillna=False` 时，在取得 `window` 个样本之前输出为 `NaN`。

## 工作原理

TII 衡量价格位于其移动平均线上方的一致程度。每根 K 线的有符号残差 \(close-SMA\) 被拆成正值部分和绝对值；对二者作滚动求和再取比率。读数接近 100，表示价格几乎始终位于 SMA 上方（强烈上涨趋势）；接近 0，表示几乎始终位于下方；接近 50，则表示上下大致均衡。

## 递推公式

令 \(c_t\) 为收盘价，\(n\) 为 `window`（默认 \(30\)）。

\[
S_t = \operatorname{SMA}_n(c_t), \qquad
d_t = c_t - S_t
\]

\[
p_t = \max(d_t, 0), \qquad
a_t = |d_t|
\]

在最近 \(\min(t,n)\) 个样本上维护滚动和：

\[
P_t = \sum_{i \in W_t} p_i, \qquad
A_t = \sum_{i \in W_t} a_i
\]

\[
TII_t = 100 \cdot \frac{P_t}{A_t}
\quad\text{（安全除法）}
\]

内部 SMA 以 `fillna=True` 构造。外层预热计数为 \(n\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class TrendIntensityIndex` 中实现，成员包括 `sma_`、`pos_` 和 `abs_`。

## 参考资料

- [TradingView：Trend Intensity Index](https://www.tradingview.com/script/uCvHH824-Trend-Intensity-Index/)
- [Investopedia：Trend](https://www.investopedia.com/terms/t/trend.asp)
