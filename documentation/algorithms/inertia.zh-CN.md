# 惯性指标（Inertia）

## 摘要

`Inertia` 是 RTTA 对 Donald Dorsey 惯性指标的流式实现：对相对波动率指数（RVI）进行线性回归。它把波动方向平滑成一个较慢的波动率趋势特征。

## 更新 API

```python
value = rtta.Inertia(
    std_window=10, smooth_window=14, reg_window=20, fillna=True
).update(close)
```

每次调用 `update(...)` 处理一个收盘价。`advance(...)` 更新状态但不返回 Python 值。标量 `batch(...)` 返回一个 NumPy 数组。

## 工作原理

Dorsey 的相对波动率指数是一种 RSI 风格的振荡器，但作用对象是收盘价的滚动标准差（上涨波动与下跌波动）。惯性指标再对该 RVI 序列应用滚动线性回归，使输出跟踪相对波动率的局部趋势，而不是原始 RVI 的每次跳动。惯性值较高，表示按 RVI 构造衡量的波动率持续偏向上涨一侧；较低则表示相反。

## 递推公式

令 \(c_t\) 为收盘价。首先使用 `std_window` 和 `smooth_window` 参数，按 `RelativeVolatilityIndex` 的定义构造 Dorsey RVI：

\[
\sigma_t = \operatorname{StdDev}_{n_\sigma}(c_t)
\]

\[
u_t =
\begin{cases}
\sigma_t & c_t > c_{t-1} \\
\tfrac12\sigma_t & c_t = c_{t-1} \\
0 & c_t < c_{t-1}
\end{cases}
\qquad
d_t =
\begin{cases}
\sigma_t & c_t < c_{t-1} \\
\tfrac12\sigma_t & c_t = c_{t-1} \\
0 & c_t > c_{t-1}
\end{cases}
\]

\[
U_t = \operatorname{EMA}_{n_s}(u_t),\quad
D_t = \operatorname{EMA}_{n_s}(d_t),\quad
\operatorname{RVI}_t = 100\cdot\frac{U_t}{U_t+D_t}
\]

随后，惯性指标是在 `reg_window` 内对 RVI 拟合的线性回归值：

\[
I_t = \operatorname{LinReg}_{n_r}(\operatorname{RVI})_t.
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class Inertia` 中实现，由 `RelativeVolatilityIndex` 与 `LinearRegressionCore` 组合而成。

## 参考资料

- [FM Labs：Relative Volatility Index](https://www.fmlabs.com/reference/default.htm?url=RVI.htm)
- [Investopedia：Relative Volatility Index](https://www.investopedia.com/terms/r/relativevolatilityindex.asp)
