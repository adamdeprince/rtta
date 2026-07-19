# MesaAdaptiveMovingAverage

## 摘要

`MesaAdaptiveMovingAverage` 是由主导周期相位驱动的 Ehlers MAMA/FAMA 自适应移动平均。

## 更新 API

```python
result = rtta.MesaAdaptiveMovingAverage().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据估计的主导周期相位调整平滑速度，以因果方式更新 MAMA 及其跟随线 FAMA。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

`update(...)` 返回含 `mama` 和 `fama` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MesaAdaptiveMovingAverage` 中实现。

## 参考资料

- [MESA Adaptive Moving Average](https://trendspider.com/learning-center/what-is-the-mesa-adaptive-moving-average-mama/)
