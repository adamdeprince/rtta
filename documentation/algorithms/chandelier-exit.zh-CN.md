# 吊灯止损（ChandelierExit）

## 摘要

`ChandelierExit` 是 RTTA 对吊灯止损指标的流式实现：分别悬挂在滚动最高价和最低价之下、之上的 ATR 跟踪型多头与空头止损位。

## 更新 API

```python
result = rtta.ChandelierExit(window=22, multiplier=3.0, fillna=True).update(
    close, high, low
)
# result.long_exit, result.short_exit
```

当 `fillna=False` 时，在取得 `window` 个样本且 ATR 可用之前，两个字段均为 `NaN`。

## 工作原理

Chuck LeBeau 提出的吊灯止损：多头止损位设在回看期最高价下方若干倍 ATR 处，空头止损位则设在回看期最低价上方若干倍 ATR 处。名称来自止损位像吊灯一样从价格极值处“悬挂”下来。当趋势方向上的极值逐步推进时，退出位随之跟踪；当价格穿越退出位时，趋势可能已经结束。

## 递推公式

令 \(C_t, H_t, L_t\) 为收盘价、最高价和最低价；\(n\) 为 `window`（默认 \(22\)）；\(m\) 为 `multiplier`（默认 \(3.0\)）。

\[
H^{\max}_t = \max_{0\le i < n} H_{t-i}, \qquad
L^{\min}_t = \min_{0\le i < n} L_{t-i}
\]

\[
A_t = \operatorname{ATR}_n(C_t, H_t, L_t)
\]

\[
\begin{aligned}
\operatorname{long\_exit}_t &= H^{\max}_t - m\, A_t \\
\operatorname{short\_exit}_t &= L^{\min}_t + m\, A_t
\end{aligned}
\]

滚动最高价使用最大极值，滚动最低价使用最小极值。内部 ATR 会继承外层的 `fillna` 标志。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ChandelierExit` 中实现，使用 `RollingExtreme` 计算最高价与最低价，并使用 `ATR`。结果字段为 `long_exit` 和 `short_exit`。另请参阅 [`ATR`](atr.zh-CN.md)。

## 参考资料

- [ChartSchool：Chandelier Exit](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chandelier-exit)
- [Investopedia：Chandelier Exit](https://www.investopedia.com/articles/trading/07/chandelier-exit.asp)
