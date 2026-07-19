# T3MovingAverage

## 摘要

`T3MovingAverage` 计算 Tillson T3 多重 EMA 移动平均。

## 更新 API

```python
result = rtta.T3MovingAverage().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式维护多层指数平滑状态，并按 T3 系数组合为当前平滑值。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class T3MovingAverage` 中实现。

## 参考资料

- [T3 Average](https://efs.kb.esignal.com/hc/en-us/articles/6362957784603-T3-Average)
