# LiquidityDroughtDetector

## 摘要

`LiquidityDroughtDetector` 以相对成交量和盘口深度识别流动性枯竭，并采用下侧回滞。

## 更新 API

```python
result = rtta.LiquidityDroughtDetector().update(volume, bid_size, ask_size)
```

`update(...)` 每次接收 `volume`、`bid_size` 和 `ask_size`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器把成交量与两侧深度相加，并相对其 EWMA 基线归一化。比率跌破进入阈值时进入枯竭状态，回升至退出阈值后离开。

## 递推公式

\[
L_t=\max(volume_t,0)+\max(bidSize_t,0)+\max(askSize_t,0)
\]

\[
q_t=\frac{L_t}{\max(B_{t-1},\epsilon)},\qquad B_t=\alpha L_t+(1-\alpha)B_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\le e\\0,&r_{t-1}=1\text{ 且 }q_t\ge x\\r_{t-1},&\text{否则}\end{cases},\qquad e<x
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class LiquidityDroughtDetector` 中实现。

## 参考资料

- [背景资料](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
