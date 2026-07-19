# SuperTrend

## 摘要

`SuperTrend` 是以 ATR 带为基础的趋势跟踪指标。

## 更新 API

```python
result = rtta.SuperTrend().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标围绕高低价中点构造 ATR 缩放带，并让当前趋势方向一侧的有效带只向价格收紧。价格穿越有效带时趋势翻转，因此该指标相当于经波动率调整的移动止损。

## 递推公式

\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t),\qquad B^+_t=\frac{high_t+low_t}{2}+mATR_t,\qquad B^-_t=\frac{high_t+low_t}{2}-mATR_t
\]

\[
U_t=\begin{cases}B^+_t,&B^+_t<U_{t-1}\text{ 或 }close_{t-1}>U_{t-1}\\U_{t-1},&\text{否则}\end{cases}
\]

\[
L_t=\begin{cases}B^-_t,&B^-_t>L_{t-1}\text{ 或 }close_{t-1}<L_{t-1}\\L_{t-1},&\text{否则}\end{cases}
\]

\[
trend_t=\begin{cases}1,&close_t\ge L_t\\-1,&close_t\le U_t\\trend_{t-1},&\text{否则}\end{cases},\qquad value_t=\begin{cases}L_t,&trend_t=1\\U_t,&trend_t=-1\end{cases}
\]

`update(...)` 返回含 `value`、`direction`、`upper` 和 `lower` 字段的结果结构体。

## 组合基础组件

[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SuperTrend` 中实现。

## 参考资料

- [背景资料：SuperTrend](https://www.investopedia.com/supertrend-indicator-7976167)
