# ParabolicSAR

## 摘要

`ParabolicSAR` 是抛物线停损转向趋势跟踪指标。

## 更新 API

```python
result = rtta.ParabolicSAR().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次更新用加速因子把 SAR 推向当前趋势的极值点。价格越过候选 SAR 时趋势反转，SAR 重置为此前极值，加速因子也从初始值重新开始。

## 递推公式

\[
SAR_t=SAR_{t-1}+AF_{t-1}(EP_{t-1}-SAR_{t-1})
\]

\[
EP_t=\begin{cases}\max(EP_{t-1},high_t),&trend_t=1\\\min(EP_{t-1},low_t),&trend_t=-1\end{cases}
\]

趋势延续并出现新极值时，加速因子逐步增大，直至设定上限。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ParabolicSAR` 中实现。

## 参考资料

- [ChartSchool：Parabolic SAR](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/parabolic-sar)
