# Twiggs 资金流量（TwiggsMoneyFlow）

## 摘要

`TwiggsMoneyFlow` 是 RTTA 对 Twiggs 资金流量的流式实现。每根 K 线的成交量根据收盘价在真实最高价/真实最低价区间中的位置确定符号；随后分别对有符号资金流与成交量作 EMA 平滑，并取二者之比。

## 更新 API

```python
value = rtta.TwiggsMoneyFlow(window=21, fillna=True).update(high, low, close, volume)
```

当 `fillna=False` 时，在取得 `window` 个样本之前输出为 `NaN`。

## 工作原理

Colin Twiggs 的资金流量指标改进了经典 Chaikin 资金流量：它使用包含前一收盘价的真实最高价和真实最低价，而不是原始最高价—最低价区间，从而减少跳空对收盘价区间位置的扭曲。它不使用简单滚动和，而是分别对累积/派发成交量与原始成交量作 EMA 平滑，得到响应更快、通常位于 \((-1,1)\) 的标准化订单流。

## 递推公式

令 \(H_t,L_t,C_t,V_t\) 为最高价、最低价、收盘价和成交量，\(n\) 为 `window`（默认 \(21\)）。第一根 K 线的真实最高价/最低价等于 \(H_t,L_t\)；此后：

\[
TH_t = \max(H_t, C_{t-1}), \qquad
TL_t = \min(L_t, C_{t-1})
\]

\[
TR_t = TH_t - TL_t
\]

\[
AD_t =
\begin{cases}
0, & TR_t = 0 \\
\dfrac{(C_t - TL_t) - (TH_t - C_t)}{TR_t}\, V_t, & \text{其他情况}
\end{cases}
\]

\[
\widetilde{AD}_t = \operatorname{EMA}_n(AD_t), \qquad
\widetilde{V}_t = \operatorname{EMA}_n(V_t)
\]

\[
TMF_t = \frac{\widetilde{AD}_t}{\widetilde{V}_t}
\quad\text{（安全除法）}
\]

两个内部 EMA 都使用 `fillna=True` 和 \(\alpha=2/(n+1)\)。外层预热计数为 \(n\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class TwiggsMoneyFlow` 中实现，成员 `ad_` 与 `vol_` 都是 `EMA`。

## 参考资料

- [Incredible Charts：Twiggs Money Flow](https://www.incrediblecharts.com/indicators/twiggs_money_flow.php)
- [ChartSchool：Chaikin Money Flow（相关指标）](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-money-flow-cmf)
