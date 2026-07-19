# CoppockCurve

## 摘要

`CoppockCurve` 对长短周期变动率之和计算加权移动平均。

## 更新 API

```python
result = rtta.CoppockCurve().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据最新收盘价更新紧凑的因果平滑状态，并返回当前估计值。

## 递推公式

令 \(z_t = close_t\) 为一次更新接收的观测。

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CoppockCurve` 中实现。

## 参考资料

- [ChartSchool：Coppock Curve](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/coppock-curve)
