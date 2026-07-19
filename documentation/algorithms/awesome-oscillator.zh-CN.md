# AwesomeOscillator

## 摘要

`AwesomeOscillator` 计算短周期与长周期中间价移动平均之差。

## 更新 API

```python
result = rtta.AwesomeOscillator().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

该指标根据最新观测更新紧凑的因果平滑状态，并返回当前振荡值。

## 递推公式

令 \(z_t = (high_t, low_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AwesomeOscillator` 中实现。

## 参考资料

- [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
