# 移动平均包络线（MovingAverageEnvelope）

## 摘要

`MovingAverageEnvelope` 是 RTTA 对简单移动平均线上下百分比包络带的流式实现。

## 更新 API

```python
result = rtta.MovingAverageEnvelope().update(value)
```

每次调用 `update(...)` 都使用 `value` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`MovingAverageEnvelope` 在简单移动平均线周围放置固定百分比的上下轨。与布林带不同，其宽度不取决于近期波动率；`percent` 是小数比例（例如 `0.025` 表示 2.5% 的包络带）。

## 递推公式

令 \(z_t=value_t\)，\(n\) 为窗口长度，\(p\) 为包络比例。

\[
M_t = \operatorname{SMA}_n(z_t)
\]

\[
U_t = M_t(1+p), \qquad
L_t = M_t(1-p)
\]

`update(...)` 返回一个结果结构体，字段为 `middle`、`upper`、`lower`。

## 组合使用的基础指标

[`SMA`](sma.zh-CN.md)

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class MovingAverageEnvelope` 中实现。

## 参考资料

- [背景资料](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-average-envelopes)
