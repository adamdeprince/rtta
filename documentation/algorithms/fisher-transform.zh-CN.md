# FisherTransform

## 摘要

`FisherTransform` 对近期高低区间内的位置应用 Ehlers 变换，得到转折点振荡器。

## 更新 API

```python
result = rtta.FisherTransform().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标维护因果的滚动极值和平滑状态，把当前价格位置归一化后映射为振荡值。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class FisherTransform` 中实现。

## 参考资料

- [Fisher Transform 指南](https://trendspider.com/learning-center/fisher-transform-a-comprehensive-guide/)
