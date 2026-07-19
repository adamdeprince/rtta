# TrendChopRegimeDetector

## 摘要

`TrendChopRegimeDetector` 以真实波幅构造效率比，区分趋势与震荡状态。

## 更新 API

```python
result = rtta.TrendChopRegimeDetector().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标用窗口首尾的净价格变化除以期间真实波幅之和。数值接近 1 表示方向性趋势，接近 0 表示来回震荡；双向回滞把该比率转换为稳定状态。

## 递推公式

\[
TR_t=\max(high_t-low_t,|high_t-close_{t-1}|,|low_t-close_{t-1}|)
\]

\[
q_t=\frac{|close_t-close_{t-n}|}{\sum_{i\in W_t}TR_i}
\]

\[
r_t=\begin{cases}1,&r_{t-1}\le0\text{ 且 }q_t\ge u_e\\0,&r_{t-1}=1\text{ 且 }q_t\le u_x\\-1,&r_{t-1}\ge0\text{ 且 }q_t\le\ell_e\\0,&r_{t-1}=-1\text{ 且 }q_t\ge\ell_x\\r_{t-1},&\text{否则}\end{cases}
\]

进入和退出常数满足 \(\ell_e < \ell_x \le u_x < u_e\)。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TrendChopRegimeDetector` 中实现。

## 参考资料

- [背景资料：Choppiness Index](https://www.angelone.in/knowledge-center/online-share-trading/choppiness-index-indicator)
