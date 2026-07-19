# VariableIndexDynamicAverage

## 摘要

`VariableIndexDynamicAverage` 是 VIDYA 自适应 EMA，以 CMO 的绝对值调整平滑系数。

## 更新 API

```python
result = rtta.VariableIndexDynamicAverage().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据近期方向性动量的强弱动态调整 EMA 响应速度，再以因果方式更新当前平均值。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VariableIndexDynamicAverage` 中实现。

## 参考资料

- [MetaTrader 5：VIDYA](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/vida)
