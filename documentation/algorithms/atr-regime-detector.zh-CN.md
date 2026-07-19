# ATRRegimeDetector

## 摘要

`ATRRegimeDetector` 是带有高低双向回滞带的有状态 ATR 市场状态检测器。

## 更新 API

```python
result = rtta.ATRRegimeDetector().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

检测器先由当前观测和紧凑的流式状态计算标量市场状态指标，再应用明确的进入/退出回滞。只有指标越过另一侧的退出阈值，输出状态才会改变。

## 递推公式

令 \(z_t = (close_t, high_t, low_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
TR_t=\max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]

\[
q_t=ATR_t=\operatorname{WilderEMA}_n(TR_t)
\]

该递推把 RTTA 的标准 `ATR` 更新与 `ThresholdRegimeDetector` 所用的双向回滞状态组合起来。

\[
r_t =
\begin{cases}
1, & r_{t-1} \le 0 \text{ 且 } q_t \ge u_e \\
0, & r_{t-1} = 1 \text{ 且 } q_t \le u_x \\
-1, & r_{t-1} \ge 0 \text{ 且 } q_t \le \ell_e \\
0, & r_{t-1} = -1 \text{ 且 } q_t \ge \ell_x \\
r_{t-1}, & \text{否则}
\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。返回值为当前标量指标值。

## 组合基础组件

[`ATR`](atr.zh-CN.md)、[`ThresholdRegimeDetector`](threshold-regime-detector.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ATRRegimeDetector` 中实现。

## 参考资料

- [ChartSchool：平均真实波幅与 ATR 百分比](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
