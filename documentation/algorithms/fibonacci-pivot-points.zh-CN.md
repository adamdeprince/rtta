# 斐波那契枢轴点（FibonacciPivotPoints）

## 摘要

`FibonacciPivotPoints` 是 RTTA 对斐波那契枢轴点组的流式实现。当前 K 线的价位由前一根 K 线的最高价、最低价和收盘价计算：以经典中央枢轴点为基准，在前一区间的 \(0.382\)、\(0.618\) 和 \(1.0\) 倍处设置支撑位和阻力位。

## 更新 API

```python
result = rtta.FibonacciPivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

第一根 K 线只用于初始化前一组 HLC。当 `fillna=True` 时，七个字段都返回该 K 线的收盘价；当 `fillna=False` 时，则全部返回 `NaN`。

## 工作原理

斐波那契枢轴点把经典中央枢轴点与应用于前一交易时段区间的斐波那契回撤比例结合起来。交易者将 R1–R3 / S1–S3 视作日内支撑位和阻力位。RTTA 只根据前一根已完成 K 线计算这些价位，随后才把此前 HLC 推进到当前 K 线——这与经典、Woodie 和 Camarilla 枢轴点采用相同的流式模式。

## 递推公式

令 \(H_{t-1},L_{t-1},C_{t-1}\) 为前一根 K 线的最高价、最低价和收盘价，并令 \(R=H_{t-1}-L_{t-1}\)。

\[
PP_t = \frac{H_{t-1} + L_{t-1} + C_{t-1}}{3}
\]

\[
\begin{aligned}
R1_t &= PP_t + 0.382\, R \\
R2_t &= PP_t + 0.618\, R \\
R3_t &= PP_t + 1.000\, R \\
S1_t &= PP_t - 0.382\, R \\
S2_t &= PP_t - 0.618\, R \\
S3_t &= PP_t - 1.000\, R
\end{aligned}
\]

输出这些价位后，RTTA 将此前 HLC 更新为当前 K 线的 \(H_t,L_t,C_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class FibonacciPivotPoints` 中实现。结果类型为 `PivotPointsResult`，字段包括 `pp`、`r1`、`r2`、`r3`、`s1`、`s2`、`s3`。

## 参考资料

- [Investopedia：Fibonacci Pivot Points](https://www.investopedia.com/articles/technical/04/041404.asp)
- [ChartSchool：Pivot Points](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/pivot-points)
