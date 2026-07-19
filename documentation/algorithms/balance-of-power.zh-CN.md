# BalanceOfPower

## 摘要

`BalanceOfPower` 根据开高低收价格衡量买方与卖方压力的相对强弱。

## 更新 API

```python
result = rtta.BalanceOfPower().update(open, high, low, close)
```

`update(...)` 每次接收 `open`、`high`、`low` 和 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

该指标维护所需的区间状态；每接收一根 K 线，C++ 实现只更新一次相应统计量。

## 递推公式

令 \(z_t = (open_t, high_t, low_t, close_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class BalanceOfPower` 中实现。

## 参考资料

- [ChartSchool：Balance of Power](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/balance-of-power-bop)
