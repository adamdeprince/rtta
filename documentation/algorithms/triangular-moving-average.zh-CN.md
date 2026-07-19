# TriangularMovingAverage

## 摘要

`TriangularMovingAverage` 计算经过两次平滑的三角移动平均。

## 更新 API

```python
result = rtta.TriangularMovingAverage().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式维护两层平滑状态，使窗口中部样本获得较高权重。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TriangularMovingAverage` 中实现。

## 参考资料

- [Triangular Moving Average](https://www.marketvolume.com/technicalanalysis/tma.asp)
