# CrossAssetCorrelationBreakDetector

## 摘要

`CrossAssetCorrelationBreakDetector` 比较两项资产的短期与长期滚动相关性，以检测相关关系失效。

## 更新 API

```python
result = rtta.CrossAssetCorrelationBreakDetector().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器在同一对数据流上同时维护短期和长期相关估计，并计算两者的绝对差。上侧回滞把该差异转换为持续的失效标志，直到两种相关估计重新收敛到退出阈值以内。

## 递推公式

\[
q_t=|\rho^{short}_t-\rho^{long}_t|
\]

短期和长期相关性由两个滚动 `Correlation` 式窗口维护。

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 组合基础组件

[`Correlation`](correlation.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CrossAssetCorrelationBreakDetector` 中实现。

## 参考资料

- [ChartSchool：相关系数](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient)
