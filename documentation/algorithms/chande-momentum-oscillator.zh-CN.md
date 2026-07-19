# ChandeMomentumOscillator

## 摘要

`ChandeMomentumOscillator` 根据近期上涨与下跌幅度之和计算动量振荡值。

## 更新 API

```python
result = rtta.ChandeMomentumOscillator().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标分别累计近期上涨和下跌幅度，再将方向性运动归一化为有界振荡值。全部状态均按因果顺序更新。

## 递推公式

令 \(z_t = close_t\) 为一次更新接收的观测。

\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}
{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ChandeMomentumOscillator` 中实现。

## 参考资料

- [Fidelity：Chande Momentum Oscillator](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo)
