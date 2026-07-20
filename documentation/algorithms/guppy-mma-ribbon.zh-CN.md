# 顾比多重移动平均带（GuppyMMARibbon）

## 摘要

`GuppyMMARibbon` 是 RTTA 对 Daryl Guppy 完整多重移动平均带的流式实现。它跟踪十二条 EMA（六条短期交易者均线和六条长期投资者均线）、两个组的平均值，以及短期平均减长期平均的价差。

## 更新 API

```python
result = rtta.GuppyMMARibbon(fillna=True).update(price)
# 短期：result.s3, s5, s8, s10, s12, s15
# 长期：result.l30, l35, l40, l45, l50, l60
# 汇总：result.short_average, result.long_average, result.spread
```

周期固定为经典的顾比参数。当 `fillna=False` 时，在取得 60 个样本（最长 EMA 周期）之前，所有字段均为 `NaN`。

## 工作原理

顾比多重移动平均把“交易者”（快速 EMA）与“投资者”（慢速 EMA）分开。当短期组紧密聚拢，并与同样方向一致的长期组拉开距离时，趋势共识很强；两组收缩或彼此交错则表示犹豫。带状形式会公开每一条 EMA 供绘图使用，而不只提供两个平均值。

RTTA 还提供精简的同类指标 [`GuppyMultipleMovingAverage`](guppy-multiple-moving-average.zh-CN.md)，只返回两个平均值及其价差。

## 递推公式

令 \(x_t\) 为价格。定义固定周期集合：

\[
\mathcal{S} = \{3,5,8,10,12,15\}, \qquad
\mathcal{L} = \{30,35,40,45,50,60\}.
\]

\[
E^{(p)}_t = \operatorname{EMA}_p(x_t)
\quad\text{对每个 } p \in \mathcal{S} \cup \mathcal{L}
\]

（RTTA 的 EMA 使用乘数 \(\alpha=2/(p+1)\)。）

\[
\begin{aligned}
S^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{S}} E^{(p)}_t \\
L^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{L}} E^{(p)}_t \\
\operatorname{spread}_t &= S^{\text{avg}}_t - L^{\text{avg}}_t
\end{aligned}
\]

命名输出一一对应：`s3` \(=E^{(3)}_t\)，…，`s15` \(=E^{(15)}_t\)；`l30` \(=E^{(30)}_t\)，…，`l60` \(=E^{(60)}_t\)，另加上述三个汇总值。内部 EMA 均以 `fillna=True` 构造，因此预热期间也有部分值；外层 `fillna=False` 时，只负责在前 59 根 K 线输出 `NaN`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class GuppyMMARibbon` 中实现。结果字段为 `s3`–`s15`、`l30`–`l60`、`short_average`、`long_average` 和 `spread`。

## 参考资料

- [Investopedia：Guppy Multiple Moving Average（GMMA）](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp)
- [ChartSchool：Guppy Multiple Moving Average](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/guppy-multiple-moving-average)
