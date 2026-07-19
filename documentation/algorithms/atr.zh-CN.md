# ATR

## 摘要

`ATR` 计算平均真实波幅（Average True Range），用于衡量 OHLC 数据流的波动性。RTTA 的实现严格遵循因果顺序：每次调用 `update(close, high, low)` 只接收一根 K 线，更新已保存的状态，并返回当前 ATR。

## 更新 API

```python
value = rtta.ATR(window=14.0, fillna=True).update(close, high, low)
```

输入为当前 K 线的收盘价、最高价和最低价。实现会保存上一收盘价、预热阶段的真实波幅累计值，以及最新的 ATR。

## 工作原理

真实波幅在单根 K 线高低价差的基础上，还计入相对上一收盘价的隔夜或跨 K 线跳空。ATR 再对真实波幅进行平滑。RTTA 在预热阶段返回截至当前所有真实波幅的算术平均；样本数达到 `window` 后，改用 Wilder 平滑。

## 递推公式

令 \(C_t\)、\(H_t\) 和 \(L_t\) 分别为第 \(t\) 次更新的收盘价、最高价和最低价，\(n\) 为 `window`。

\[
TR_t =
\begin{cases}
H_t - L_t, & t = 0 \\
\max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|), & t > 0
\end{cases}
\]

预热阶段，令 \(m_t = t + 1 \le n\)：

\[
ATR_t = \frac{\sum_{i=0}^{t} TR_i}{m_t}
\]

预热结束后：

\[
ATR_t = \frac{(n - 1)ATR_{t-1} + TR_t}{n}
\]

当 `fillna=False` 时，对象在预热阶段仍会更新全部内部状态，但在接收至少 `window` 个样本之前返回 `NaN`。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ATR` 中实现。

## 参考资料

- [ChartSchool：平均真实波幅与 ATR 百分比](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
