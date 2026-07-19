# EhlersOptimalTrackingFilter

## 摘要

`EhlersOptimalTrackingFilter` 使用 Ehlers 价格不确定性跟踪指数进行自适应跟踪。

## 更新 API

```python
result = rtta.EhlersOptimalTrackingFilter().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次调用只接收一组新的高低价观测，先按价格不确定性调整内部状态，再返回当前滤波值。

## 递推公式

\[
s_t=F_{EhlersOptimalTrackingFilter}(s_{t-1},(high_t,low_t);\theta)
\]

\[
y_t=G_{EhlersOptimalTrackingFilter}(s_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class EhlersOptimalTrackingFilter` 中实现。

## 参考资料

- [John Ehlers Optimal Tracking Filter](https://www.prorealcode.com/prorealtime-indicators/john-ehlers-optimal-tracking-filter/)
