# ATRP

## 摘要

`ATRP` 以当前价格的百分比表示平均真实波幅。

## 更新 API

```python
result = rtta.ATRP().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

`ATRP` 用当前收盘价对 `ATR` 波动估计进行归一化，使以价格单位计量的波幅可以跨价格水平比较，同时保留 `ATR` 的单步 Wilder 平滑。

## 递推公式

令 \(z_t = (close_t, high_t, low_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t)
\]

\[
y_t=\frac{ATR_t}{close_t}
\]

返回值为当前标量指标值。

## 组合基础组件

[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ATRP` 中实现。

## 参考资料

- [ChartSchool：平均真实波幅与 ATR 百分比](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
