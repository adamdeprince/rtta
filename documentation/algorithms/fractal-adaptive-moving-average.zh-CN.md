# FractalAdaptiveMovingAverage

## 摘要

`FractalAdaptiveMovingAverage` 是 Ehlers FRAMA，以分形维数自适应调整 EMA 平滑速度。

## 更新 API

```python
result = rtta.FractalAdaptiveMovingAverage(window=16).update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据近期价格路径的分形维数调整平滑系数，再以因果方式更新当前平均值。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class FractalAdaptiveMovingAverage` 中实现。

## 参考资料

- [MetaTrader 5：FRAMA](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama)
