# MicrostructureNoiseRegimeDetector

## 摘要

`MicrostructureNoiseRegimeDetector` 用报价价差归一化成交价相对中点的偏离，并以 EWMA 检测微观结构噪声状态。

## 更新 API

```python
result = rtta.MicrostructureNoiseRegimeDetector().update(trade_price, bid_price, ask_price)
```

`update(...)` 每次接收成交价、买价和卖价；只推进状态时可调用 `advance(...)`。

## 工作原理

成交价相对报价中点的绝对偏离先除以当时价差，再进行 EWMA 平滑。回滞逻辑把平滑指标转换为稳定的高噪声状态。

## 递推公式

\[
mid_t=\frac{bid_t+ask_t}{2},\qquad n_t=\frac{|trade_t-mid_t|}{\max(ask_t-bid_t,\epsilon)}
\]

\[
q_t=\alpha n_t+(1-\alpha)q_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MicrostructureNoiseRegimeDetector` 中实现。

## 参考资料

- [背景资料：市场微观结构](https://en.wikipedia.org/wiki/Market_microstructure)
