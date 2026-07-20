# Bias

## 摘要

`Bias` 是 RTTA 对价格偏离其简单移动平均线幅度的流式百分比计算。它以该平均值
为基准，衡量当前收盘价高于或低于 SMA 的百分比。

## 更新 API

```python
value = rtta.Bias(window=20, fillna=True).update(close)
```

每次 `update(...)` 输入一个收盘价。当 `fillna=False` 时，在 SMA 窗口填满
之前输出 `NaN`，其预热策略与内部 SMA 相同。

## 工作原理

乖离率（Bias，在亚洲市场中较为常用，也称“百分比乖离率”或“价格乖离率”）是
价格与移动平均线之间的标准化距离。正值表示价格向上偏离均值，负值表示价格位于
均值下方。由于分母是平均值本身，乖离率不受价格尺度影响，可以在不同标的之间
比较。

RTTA 通过内部 SMA，按照 \(100\times(close-SMA)/SMA\) 计算 Bias。

## 递推公式

令 \(c_t\) 为收盘价，\(n\) 为 `window`（默认 \(20\)）。

\[
S_t = \operatorname{SMA}_n(c_t)
\]

\[
\operatorname{Bias}_t = 100 \cdot \frac{c_t - S_t}{S_t}
\quad\text{（安全除法）}
\]

当 \(S_t=0\) 或不是有限值时，安全除法路径会按照 RTTA 的 `safe_divide`
辅助函数约定返回 `0` 或 `NaN`。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class Bias` 中。内部 SMA 成员
使用与外层指标相同的 `fillna` 设置。

## 参考资料

- [Investopedia：Moving Average](https://www.investopedia.com/terms/m/movingaverage.asp)
- [TradingView：Bias 指标概念](https://www.tradingview.com/script/kI9g2u0a-Bias/)
