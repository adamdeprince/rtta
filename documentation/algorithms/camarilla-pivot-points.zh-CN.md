# CamarillaPivotPoints

## 摘要

`CamarillaPivotPoints` 是 RTTA 对 Camarilla 枢轴点组的流式实现。当前 K 线
的各级价位根据前一根 K 线的最高价、最低价和收盘价，使用 Camarilla 区间倍数
计算；随后当前 K 线成为下一次更新所使用的前一根 K 线。

## 更新 API

```python
result = rtta.CamarillaPivotPoints(fillna=True).update(high, low, close)
# result.pp, result.r1, result.r2, result.r3, result.s1, result.s2, result.s3
```

第一根 K 线只用于初始化前一根 HLC。当 `fillna=True` 时，七个字段都返回该
K 线的收盘价；当 `fillna=False` 时则全部返回 `NaN`。

## 工作原理

Nick Scott 的 Camarilla 方程使用前一根 K 线区间的固定比例，在前收盘价附近
紧密设置支撑位和阻力位。经典价位从 R1/S1 到 R3/S3，分别使用
\(1.1/12\)、\(1.1/6\) 和 \(1.1/4\) 倍数。为便于使用，RTTA 还在与经典、
Woodie 和 Fibonacci 枢轴点相同的结果结构体中，返回传统中枢价
\(PP=(H+L+C)/3\)。

由于各价位只依赖前一根已经完成的 K 线，流式实现会保存前一根 HLC，在覆盖这组
状态之前，每根 K 线重新计算一次价位。

## 递推公式

令 \(H_{t-1},L_{t-1},C_{t-1}\) 分别为前一根 K 线的最高价、最低价和收盘价，
并令 \(R=H_{t-1}-L_{t-1}\)。

\[
PP_t = \frac{H_{t-1} + L_{t-1} + C_{t-1}}{3}
\]

\[
\begin{aligned}
R1_t &= C_{t-1} + R \cdot \tfrac{1.1}{12} \\
R2_t &= C_{t-1} + R \cdot \tfrac{1.1}{6} \\
R3_t &= C_{t-1} + R \cdot \tfrac{1.1}{4} \\
S1_t &= C_{t-1} - R \cdot \tfrac{1.1}{12} \\
S2_t &= C_{t-1} - R \cdot \tfrac{1.1}{6} \\
S3_t &= C_{t-1} - R \cdot \tfrac{1.1}{4}
\end{aligned}
\]

输出这些价位后，RTTA 将前一根 HLC 更新为当前 K 线的 \(H_t,L_t,C_t\)。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class CamarillaPivotPoints`
中。结果类型为 `PivotPointsResult`，字段包括 `pp`、`r1`、`r2`、`r3`、
`s1`、`s2` 和 `s3`。

## 参考资料

- [Investopedia：Camarilla Pivot Points](https://www.investopedia.com/terms/c/camarilla-pivot-point.asp)
- [TradingPedia：Camarilla Pivot Points](https://www.tradingpedia.com/forex-trading-strategies/camarilla-pivot-points/)
