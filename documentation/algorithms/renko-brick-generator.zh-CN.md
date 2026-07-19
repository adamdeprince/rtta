# RenkoBrickGenerator

## 摘要

`RenkoBrickGenerator` 是事件驱动的 Renko 价格变换，根据收盘价更新输出带符号的砖块数量和当前砖块状态。

## 更新 API

```python
result = rtta.RenkoBrickGenerator().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

当前收盘价相对锚点每跨越一个 `brick_size` 就生成一块砖；一次更新可以生成多块，方向由带符号的砖块数决定。

## 递推公式

\[
k_t=\left\lfloor\frac{close_t-anchor_{t-1}}{brick\_size}\right\rfloor
\]

\[
anchor_t=anchor_{t-1}+k_t\,brick\_size,\qquad direction_t=\operatorname{sgn}(k_t)
\]

`update(...)` 返回含 `brick_open`、`brick_close`、`direction`、`bricks` 和 `reversal` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RenkoBrickGenerator` 中实现。

## 参考资料

- [TradingView：Renko 图](https://www.tradingview.com/support/solutions/43000502284-understanding-renko-charts/)
