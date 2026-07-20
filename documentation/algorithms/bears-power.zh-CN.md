# BearsPower

## 摘要

`BearsPower` 是 RTTA 对 Alexander Elder 空头力量指标的流式实现：计算每根
K 线最低价低于收盘价 EMA 的距离。它是 `ElderRayIndex` 的空头分量，并以单一
标量形式提供给常见零售交易 API。

## 更新 API

```python
value = rtta.BearsPower(window=13, fillna=True).update(low, close)
```

`update(...)` 接收 `low` 和 `close`。`advance(...)` 使用相同输入，但不返回
Python 值。标量批量接口 `batch(low, close)` 返回 NumPy 数组。

## 工作原理

Elder 用最低价相对于收盘价共识值（EMA）的下探距离来刻画“空头”。空头力量为
较大的负值，意味着最低价远低于平滑后的收盘价；接近零则表示最低价紧贴 EMA。
它与 `BullsPower` 一起构成双向的 Elder-ray 多空压力视图。

## 递推公式

令 \(c_t\) 为收盘价、\(\ell_t\) 为最低价、\(n\) 为 `window`。令
\(\operatorname{EMA}_n\) 表示 RTTA 对收盘价计算的指数移动平均。

\[
E_t = \operatorname{EMA}_n(c_t)
\]

\[
\operatorname{Bears}_t = \ell_t - E_t
\]

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class BearsPower` 中。采用相同
EMA 初始化路径时，其结果与 `ElderRayIndex.bear_power` 一致。

## 参考资料

- [Investopedia：Elder-Ray Index](https://www.investopedia.com/articles/trading/03/022603.asp)
- [ChartSchool：Elder-Ray Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/elder-ray-index)
