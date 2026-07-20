# Elder 温度计（ElderThermometer）

## 摘要

`ElderThermometer` 是 RTTA 对 Elder 风格 K 线区间温度计的流式实现。每根 K 线都会报告其最高价—最低价区间、该区间与前一根 K 线区间之比，以及当前区间超过前一区间时取真的二元“升温”标志。

## 更新 API

```python
result = rtta.ElderThermometer(fillna=True).update(high, low)
# result.ratio, result.hot, result.range
```

若 `high < low`，实现会交换二者，确保区间始终非负。第一根 K 线用于初始化前一区间；当 `fillna=True` 时，返回 `ratio=1.0`、`hot=0.0` 和当前区间；当 `fillna=False` 时，第一根 K 线的所有字段均为 `NaN`。

## 工作原理

Alexander Elder 用市场“温度计”的概念比较今日与昨日的活动区间。区间扩张（“升温”的 K 线）常伴随趋势行情或新闻驱动的交易时段；区间收缩则表示市场更为平静。RTTA 提供三个相关输出，调用方可以对比率设阈值、使用二元升温标志，或直接绘制原始区间。

## 递推公式

令 \(H_t,L_t\) 为最高价和最低价（必要时交换，使 \(H_t\ge L_t\)）。

\[
\rho_t = H_t - L_t
\]

第一根 K 线把 \(\rho_t\) 保存为前一区间，并输出 `fillna` 占位值。对于 \(t\ge1\)：

\[
\operatorname{ratio}_t = \frac{\rho_t}{\rho_{t-1}}
\quad\text{（安全除法）}
\]

\[
\operatorname{hot}_t =
\begin{cases}
1, & \rho_t > \rho_{t-1} \\
0, & \text{其他情况}
\end{cases}
\]

\[
\operatorname{range}_t = \rho_t
\]

随后将 \(\rho_t\) 设为新的前一区间。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ElderThermometer` 中实现。结果字段为 `ratio`、`hot` 和 `range`。

## 参考资料

- [Investopedia：Alexander Elder](https://www.investopedia.com/terms/a/alexander-elder.asp)
- [TradingView：Elder Thermometer 相关概念](https://www.tradingview.com/scripts/elder/)
