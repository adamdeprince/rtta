# VolumePriceTrend

## 摘要

`VolumePriceTrend` 按百分比价格变化调整成交量并进行累计。

## 更新 API

```python
result = rtta.VolumePriceTrend().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标把每期成交量乘以价格百分比变化后加入累计值，更新只依赖最新 tick 与此前状态。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t,\qquad V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolumePriceTrend` 中实现。

## 参考资料

- [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
