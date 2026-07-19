# SpreadFeatures

## 摘要

`SpreadFeatures` 根据成交及其同时刻报价估计报价价差、有效价差和已实现价差。

## 更新 API

```python
result = rtta.SpreadFeatures().update(trade_price, bid_price, ask_price)
```

`update(...)` 每次接收成交价、买价和卖价；只推进状态时可调用 `advance(...)`。

## 工作原理

实现把成交价与报价中点、报价宽度组合成流式市场微观结构特征，更新只依赖最新 tick 与此前状态。

## 递推公式

\[
mid_t=\frac{bid_t+ask_t}{2},\qquad spread_t=\max(ask_t-bid_t,0)
\]

\[
relative\_spread_t=\frac{spread_t}{\max(|mid_t|,\epsilon)},\qquad trade\_location_t=\frac{trade_t-mid_t}{\max(spread_t,\epsilon)}
\]

`update(...)` 返回含 `quoted_spread`、`effective_spread` 和 `realized_spread` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SpreadFeatures` 中实现。

## 参考资料

- [背景资料](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
