# DeMarker 指标

## 摘要

`DeMarker` 是 RTTA 对 DeMarker 振荡器的流式实现。它比较滚动的最高价向上延伸量与最低价向下延伸量，并将二者映射为 \(0\)–\(1\) 之间的比率。

## 更新 API

```python
value = rtta.DeMarker(window=14, fillna=True).update(high, low)
```

第一根 K 线的 DeMax/DeMin 贡献均为零（因为没有前一根最高价/最低价）。当 `fillna=False` 时，在窗口缓冲区填满之前，输出为 `NaN`。

## 工作原理

Tom DeMark 的 DeMarker 以最高价之间的正向变化衡量需求压力，以最低价之间的负向变化衡量供给压力。分别在窗口内求平均再取比率，便得到取值范围为 \([0,1]\) 的振荡器。读数接近 \(1\) 表示最高价持续向上延伸，接近 \(0\) 表示最低价持续向下延伸。极值区（经典取值约为 \(0.7/0.3\)）常作为超买/超卖参考。

## 递推公式

令 \(H_t, L_t\) 为最高价和最低价，\(n\) 为 `window`（默认 \(14\)）。

\[
DeMax_t = \max(H_t - H_{t-1}, 0), \qquad
DeMin_t = \max(L_{t-1} - L_t, 0)
\]

（其中 \(DeMax_0 = DeMin_0 = 0\)）。维护最近 \(\min(t+1,n)\) 个样本的滚动和：

\[
M_t = \sum_{i \in W_t} DeMax_i, \qquad
N_t = \sum_{i \in W_t} DeMin_i
\]

\[
DeM_t = \frac{M_t}{M_t + N_t}
\quad\text{（安全除法）}
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class DeMarker` 中实现，使用两个滚动求和缓冲区（`demax_`、`demin_`）。

## 参考资料

- [Investopedia：DeMarker Indicator](https://www.investopedia.com/terms/d/demarkerindicator.asp)
- [TradingPedia：DeMarker](https://www.tradingpedia.com/forex-trading-indicators/demarker-indicator/)
