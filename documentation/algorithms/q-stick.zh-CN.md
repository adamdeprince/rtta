# QStick 指标

## 摘要

`QStick` 是 RTTA 对 QStick 指标的流式实现：收盘价减开盘价所得 K 线实体的简单移动平均。正值表示近期 K 线大多看涨，负值表示大多看跌。

## 更新 API

```python
value = rtta.QStick(window=14, fillna=True).update(open, close)
```

内部 SMA 会继承 `fillna`。

## 工作原理

Tushar Chande 的 QStick 汇总近期 K 线有符号实体的平均值。与纯收盘价动量不同，它使用每根 K 线的开盘到收盘变动，因此即使价格区间很大，只要收盘接近开盘，贡献也很小。零轴交叉可视作短期趋势变化；极值则可标记衰竭。

## 递推公式

令 \(O_t,C_t\) 为开盘价和收盘价，\(n\) 为 `window`（默认 \(14\)）。

\[
b_t = C_t - O_t
\]

\[
QStick_t = \operatorname{SMA}_n(b_t)
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class QStick` 中实现。唯一成员是一个以 `close - open` 为输入的 `SMA`。

## 参考资料

- [Investopedia：Qstick Indicator](https://www.investopedia.com/terms/q/qstick.asp)
- [TradingPedia：QStick](https://www.tradingpedia.com/forex-trading-indicators/qstick-indicator/)
