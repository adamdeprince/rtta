# 零滞后指数移动平均（ZeroLagEMA）

## 摘要

`ZeroLagEMA` 是 RTTA 对零滞后指数移动平均的流式实现：先对价格去滞后，再送入 EMA。

## 更新 API

```python
result = rtta.ZeroLagEMA().update(value)
```

每次调用 `update(...)` 都使用 `value` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`ZeroLagEMA` 通过把去滞后序列送入标准 EMA 来减少 EMA 滞后。估计滞后量为 EMA 周期的一半（整数），去滞后输入等于当前价格加上当前价格与滞后价格之差。

## 递推公式

令 \(z_t=value_t\)，\(n\) 为窗口长度，\(L=\lfloor(n-1)/2\rfloor\)。

\[
\tilde{z}_t = 2z_t - z_{t-L}
\]

\[
y_t = \operatorname{EMA}_n(\tilde{z}_t)
\]

EMA 初始化规则与 [`EMA`](ema.zh-CN.md) 相同。当 \(L=0\) 时，\(\tilde z_t=z_t\)。

返回值为当前的标量指标值。

## 组合使用的基础指标

[`EMA`](ema.zh-CN.md)

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ZeroLagEMA` 中实现。

## 参考资料

- [背景资料](https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average)
