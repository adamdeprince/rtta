# SchaffTrendCycle

## 摘要

`SchaffTrendCycle` 将 MACD 与随机周期归一化结合为振荡器。

## 更新 API

```python
result = rtta.SchaffTrendCycle().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式维护快慢趋势和平滑周期状态，并映射为当前振荡值。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SchaffTrendCycle` 中实现。

## 参考资料

- [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
