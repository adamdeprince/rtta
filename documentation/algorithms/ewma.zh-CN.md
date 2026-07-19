# EWMA

## 摘要

`EWMA` 计算由 alpha、span 或 com 参数化的指数加权移动平均。

## 更新 API

```python
result = rtta.EWMA(span=30.0).update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

最新观测以固定权重进入指数状态，此前估计以互补权重保留；因此每次更新均为常数时间。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class EWMA` 中实现。

## 参考资料

- [pandas：指数加权窗口](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
