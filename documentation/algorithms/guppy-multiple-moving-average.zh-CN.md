# 顾比多重移动平均（GuppyMultipleMovingAverage）

## 摘要

`GuppyMultipleMovingAverage` 是 RTTA 对顾比多重移动平均（MMA）的精简流式实现：每根 K 线更新六条短期 EMA 和六条长期 EMA，再汇总为短期组平均、长期组平均及二者价差。

## 更新 API

```python
result = rtta.GuppyMultipleMovingAverage(fillna=True).update(price)
# result.short_average, result.long_average, result.spread
```

周期固定：短期为 \(\{3,5,8,10,12,15\}\)，长期为 \(\{30,35,40,45,50,60\}\)。当 `fillna=False` 时，在取得 60 个样本之前输出为 `NaN`。

## 工作原理

Daryl Guppy 的多重移动平均把市场参与者划分为两类：短期交易者与长期投资者。分别对两组 EMA 取平均可得到两条汇总曲线；二者之间的距离（`spread`）是衡量趋势强度 / 共识程度的简洁指标。如需完整的十二线带状图，请使用 [`GuppyMMARibbon`](guppy-mma-ribbon.zh-CN.md)。

## 递推公式

令 \(x_t\) 为价格，

\[
\mathcal{S} = \{3,5,8,10,12,15\}, \qquad
\mathcal{L} = \{30,35,40,45,50,60\}.
\]

\[
\begin{aligned}
S^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{S}} \operatorname{EMA}_p(x_t) \\
L^{\text{avg}}_t &= \tfrac16 \sum_{p\in\mathcal{L}} \operatorname{EMA}_p(x_t) \\
\operatorname{spread}_t &= S^{\text{avg}}_t - L^{\text{avg}}_t
\end{aligned}
\]

每个 \(\operatorname{EMA}_p\) 都像 RTTA 的 `class EMA` 一样使用 \(\alpha=2/(p+1)\)。内部 EMA 在预热期间始终填充值；外层 `fillna=False` 时，只把前 59 个样本置为 `NaN`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class GuppyMultipleMovingAverage` 中实现。结果字段为 `short_average`、`long_average` 和 `spread`。

## 参考资料

- [Investopedia：Guppy Multiple Moving Average（GMMA）](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp)
- [ChartSchool：Guppy Multiple Moving Average](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/guppy-multiple-moving-average)
