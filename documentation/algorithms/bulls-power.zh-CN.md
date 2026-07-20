# BullsPower

## 摘要

`BullsPower` 是 RTTA 对 Alexander Elder 多头力量指标的流式实现：计算每根
K 线最高价高于收盘价 EMA 的距离。它是 `ElderRayIndex` 的多头分量，并以单一
标量形式提供给常见零售交易 API。

## 更新 API

```python
value = rtta.BullsPower(window=13, fillna=True).update(high, close)
```

`update(...)` 接收 `high` 和 `close`。`advance(...)` 使用相同输入，但不返回
Python 值。标量批量接口 `batch(high, close)` 返回 NumPy 数组。

## 工作原理

Elder 用最高价相对于收盘价共识值（EMA）的上冲距离来刻画“多头”。较大的正值
意味着最高价远高于平滑后的收盘价；接近零则表示最高价被压在 EMA 附近。它与
`BearsPower`（或 `ElderRayIndex`）结合，构成经典 Elder-ray 的多空压力
视图。

## 递推公式

令 \(c_t\) 为收盘价、\(h_t\) 为最高价、\(n\) 为 `window`。令
\(\operatorname{EMA}_n\) 表示 RTTA 对收盘价计算的指数移动平均。

\[
E_t = \operatorname{EMA}_n(c_t)
\]

\[
\operatorname{Bulls}_t = h_t - E_t
\]

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class BullsPower` 中。采用相同
EMA 初始化路径时，其结果与 `ElderRayIndex.bull_power` 一致。

## 参考资料

- [Investopedia：Elder-Ray Index](https://www.investopedia.com/articles/trading/03/022603.asp)
- [ChartSchool：Elder-Ray Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/elder-ray-index)
