# OrderFlowImbalance

## 摘要

`OrderFlowImbalance` 在滚动更新窗口内衡量最优买卖价及挂单量变化所产生的压力。

## 更新 API

```python
result = rtta.OrderFlowImbalance().update(bid_price, bid_size, ask_price, ask_size)
```

`update(...)` 每次接收最优买卖价及其挂单量；只推进状态时可调用 `advance(...)`。

## 工作原理

实现把每次盘口更新推入滚动窗口，并从价格和挂单量的变化中累计买卖双方压力。

## 递推公式

\[
W_t=\operatorname{push}(W_{t-1},z_t,n),\qquad y_t=G(W_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class OrderFlowImbalance` 中实现。

## 参考资料

- [订单流失衡论文](https://arxiv.org/abs/1011.6402)
