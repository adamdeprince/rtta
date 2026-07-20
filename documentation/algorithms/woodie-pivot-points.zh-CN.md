# Woodie 枢轴点（WoodiePivotPoints）

## 摘要

`WoodiePivotPoints` 是 RTTA 对 Woodie 枢轴点组的流式实现。当前 K 线的价位由前一根 K 线的最高价、最低价和收盘价计算，其中收盘价权重加倍：\(PP=(H+L+2C)/4\)。

## 更新 API

```python
result = rtta.WoodiePivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

第一根 K 线只用于初始化此前 HLC。当 `fillna=True` 时，七个字段都返回该 K 线的收盘价；当 `fillna=False` 时，则全部返回 `NaN`。

## 工作原理

Woodie 枢轴点对前一收盘价赋予的权重高于经典场内枢轴点。随后，根据该枢轴和前一区间导出支撑位与阻力位，R1/S1/R2/S2 公式与经典形式相似；R3/S3 则采用与 RTTA 经典 `PivotPoints` 相同的延伸模式。

## 递推公式

令 \(H_{t-1},L_{t-1},C_{t-1}\) 为前一根 K 线的最高价、最低价和收盘价，并令 \(R=H_{t-1}-L_{t-1}\)。

\[
PP_t = \frac{H_{t-1} + L_{t-1} + 2 C_{t-1}}{4}
\]

\[
\begin{aligned}
R1_t &= 2\, PP_t - L_{t-1} \\
S1_t &= 2\, PP_t - H_{t-1} \\
R2_t &= PP_t + R \\
S2_t &= PP_t - R \\
R3_t &= H_{t-1} + 2\,(PP_t - L_{t-1}) \\
S3_t &= L_{t-1} - 2\,(H_{t-1} - PP_t)
\end{aligned}
\]

输出这些价位后，RTTA 将此前 HLC 更新为当前 K 线的 \(H_t,L_t,C_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class WoodiePivotPoints` 中实现。结果类型为 `PivotPointsResult`，字段包括 `pp`、`r1`、`r2`、`r3`、`s1`、`s2`、`s3`。

## 参考资料

- [Investopedia：Woodie Pivot Points](https://www.investopedia.com/articles/technical/04/041404.asp)
- [ChartSchool：Pivot Points](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/pivot-points)
