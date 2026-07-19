# ChaikinMoneyFlow

## 摘要

`ChaikinMoneyFlow` 计算指定窗口内的成交量加权资金流。

## 更新 API

```python
result = rtta.ChaikinMoneyFlow().update(close, high, low, volume)
```

`update(...)` 每次接收 `close`、`high`、`low` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据收盘价在当根 K 线区间中的位置确定资金流乘数，再以成交量加权并在窗口内累计。

## 递推公式

令 \(z_t = (close_t, high_t, low_t, volume_t)\) 为一次更新接收的观测。

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ChaikinMoneyFlow` 中实现。

## 参考资料

- [ChartSchool：Chaikin 资金流](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-money-flow-cmf)
