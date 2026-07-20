# 枢轴点（PivotPoints）

## 摘要

`PivotPoints` 是 RTTA 对经典场内交易员枢轴点组的流式实现。当前 K 线的价位由前一根 K 线的最高价、最低价和收盘价计算；随后，当前 K 线成为下一次计算的前一根 K 线。

## 更新 API

```python
result = rtta.PivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

第一根 K 线只用于初始化此前 HLC。当 `fillna=True` 时，七个字段都返回该 K 线的收盘价；当 `fillna=False` 时，则全部返回 `NaN`。

## 工作原理

经典枢轴点把前一交易时段的最高价、最低价和收盘价平均为中央枢轴 \(PP\)，再根据该枢轴和此前价格区间向外投射阻力位与支撑位。场内交易员历来把它们作为日内参考价格。RTTA 的流式形式使用逐根 K 线的此前 HLC（适用于任意规则 K 线流），而不是单独的交易时段日历。

相关变体：[`WoodiePivotPoints`](woodie-pivot-points.zh-CN.md)、[`CamarillaPivotPoints`](camarilla-pivot-points.zh-CN.md)、[`FibonacciPivotPoints`](fibonacci-pivot-points.zh-CN.md)。

## 递推公式

令 \(H_{t-1},L_{t-1},C_{t-1}\) 为前一根 K 线的最高价、最低价和收盘价。

\[
PP_t = \frac{H_{t-1} + L_{t-1} + C_{t-1}}{3}
\]

\[
\begin{aligned}
R1_t &= 2\, PP_t - L_{t-1} \\
S1_t &= 2\, PP_t - H_{t-1} \\
R2_t &= PP_t + (H_{t-1} - L_{t-1}) \\
S2_t &= PP_t - (H_{t-1} - L_{t-1}) \\
R3_t &= H_{t-1} + 2\,(PP_t - L_{t-1}) \\
S3_t &= L_{t-1} - 2\,(H_{t-1} - PP_t)
\end{aligned}
\]

输出这些价位后，RTTA 将此前 HLC 更新为当前 K 线的 \(H_t,L_t,C_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class PivotPoints` 中实现。结果类型为 `PivotPointsResult`，字段包括 `pp`、`r1`、`r2`、`r3`、`s1`、`s2`、`s3`。

## 参考资料

- [ChartSchool：Pivot Points](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/pivot-points)
- [Investopedia：Pivot Point](https://www.investopedia.com/terms/p/pivotpoint.asp)
