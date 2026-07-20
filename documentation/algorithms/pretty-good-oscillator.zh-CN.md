# Pretty Good 振荡器（PrettyGoodOscillator）

## 摘要

`PrettyGoodOscillator` 是 RTTA 对 Mark Johnson Pretty Good Oscillator（PGO）的流式实现：收盘价减去其 SMA，再用同一窗口的 ATR 标准化。

## 更新 API

```python
value = rtta.PrettyGoodOscillator(window=14, fillna=True).update(close, high, low)
```

当 `fillna=False` 时，在取得 `window` 个样本之前输出为 `NaN`（ATR 的 `fillna` 也会生效）。

## 工作原理

PGO 衡量收盘价距其简单均值有多少个 ATR。正值表示市场向上偏离平均线，负值表示向下偏离。\(\pm3\) 等阈值有时用于判断衰竭或构建均值回归交易。由于 ATR 始终非负，PGO 的符号只由 \(close-SMA\) 决定。

## 递推公式

令 \(C_t,H_t,L_t\) 为收盘价、最高价和最低价，\(n\) 为 `window`（默认 \(14\)）。

\[
S_t = \operatorname{SMA}_n(C_t), \qquad
A_t = \operatorname{ATR}_n(C_t, H_t, L_t)
\]

\[
PGO_t = \frac{C_t - S_t}{A_t}
\quad\text{（安全除法）}
\]

内部 SMA 以 `fillna=True` 构造，因此预热期间可以得到部分均值；ATR 使用外层 `fillna` 标志。外层预热计数为 \(n\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class PrettyGoodOscillator` 中实现。另请参阅 [`ATR`](atr.zh-CN.md)。

## 参考资料

- [TradingView：Pretty Good Oscillator（PGO）](https://www.tradingview.com/script/rNYgL8uA-Pretty-Good-Oscillator-PGO/)
- [Investopedia：Average True Range（ATR）](https://www.investopedia.com/terms/a/atr.asp)
