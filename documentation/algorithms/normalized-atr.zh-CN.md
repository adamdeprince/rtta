# NormalizedATR

## 摘要

`NormalizedATR` 用收盘价对 ATR 进行归一化。

## 更新 API

```python
result = rtta.NormalizedATR().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以当前收盘价表示 ATR，使价格单位的波幅可以跨价格水平比较，同时保留 Wilder 真实波幅平滑。

## 递推公式

\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t),\qquad y_t=\frac{ATR_t}{close_t}
\]

## 组合基础组件

[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class NormalizedATR` 中实现。

## 参考资料

- [ChartSchool：ATR 与 ATR 百分比](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
