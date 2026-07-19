# RollingSpreadLiquidityShiftDetector

## 摘要

`RollingSpreadLiquidityShiftDetector` 以相邻窗口的报价价差/深度压力检测流动性位移。

## 更新 API

```python
result = rtta.RollingSpreadLiquidityShiftDetector(window=20, threshold=1e-06).update(bid_price, bid_size, ask_price, ask_size)
```

`update(...)` 每次接收最优买卖价及挂单量；只推进状态时可调用 `advance(...)`。

## 工作原理

每条盘口观测先转换为单位深度的价差压力，再比较参考窗口与近期窗口的均值；差异越过阈值时输出方向。

## 递推公式

\[
s_t=\frac{\max(ask_t-bid_t,0)}{\max(bid\_size_t+ask\_size_t,\epsilon)},\qquad q_t=\operatorname{mean}(R^s_t)-\operatorname{mean}(B^s_t)
\]

\[
r_t=\begin{cases}1,&q_t>h\\-1,&q_t<-h\\0,&\text{否则}\end{cases}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RollingSpreadLiquidityShiftDetector` 中实现。

## 参考资料

- [订单流失衡论文](https://arxiv.org/abs/1011.6402)
