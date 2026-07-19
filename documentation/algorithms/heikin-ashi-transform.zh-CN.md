# HeikinAshiTransform

## 摘要

`HeikinAshiTransform` 增量计算 Heikin-Ashi OHLC，用于平滑 K 线。

## 更新 API

```python
result = rtta.HeikinAshiTransform().update(open, high, low, close)
```

`update(...)` 每次接收一根 K 线的开高低收；只推进状态时可调用 `advance(...)`。

## 工作原理

每次调用只接收一组新观测，先由当前和上一根 Heikin-Ashi K 线推进状态，再返回转换后的 OHLC。

## 递推公式

\[
HAclose_t=\frac{open_t+high_t+low_t+close_t}{4}
\]

\[
HAopen_t=\frac{HAopen_{t-1}+HAclose_{t-1}}2
\]

\[
HAhigh_t=\max(high_t,HAopen_t,HAclose_t),\qquad HAlow_t=\min(low_t,HAopen_t,HAclose_t)
\]

`update(...)` 返回含 `open`、`high`、`low` 和 `close` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class HeikinAshiTransform` 中实现。

## 参考资料

- [Heikin-Ashi 背景资料](https://www.mql5.com/en/articles/19260)
