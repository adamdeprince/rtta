# RSI

## 摘要

`RSI` 对单一标量价格流增量计算相对强弱指数。它保存前一个价格，以及持续更新的上涨与下跌幅度状态。

## 更新 API

```python
value = rtta.RSI(window=14, fillna=True).update(value)
```

第一个样本用于初始化 `prev`。当 `fillna=True` 时，第一次返回 `50.0`；当 `fillna=False` 时，预热阶段返回 `NaN`。

## 工作原理

RSI 比较近期上涨幅度与下跌幅度，并把两者的比率映射为有界振荡值。RTTA 的 C++ 实现在价格上涨时更新上涨状态，在价格下跌时更新下跌状态。预热阶段以当前样本数为平均分母；预热结束后，按 `window` 采用 Wilder 平滑。

## 递推公式

令 \(x_t\) 为当前值，\(x_{t-1}\) 为前一值，\(n\) 为 `window`。

\[
g_t = \max(x_t - x_{t-1}, 0), \qquad
\ell_t = \max(x_{t-1} - x_t, 0)
\]

预热阶段，RTTA 只更新实际发生变动的一侧：

\[
G_t =
\begin{cases}
\frac{(t-1)G_{t-1}+g_t}{t}, & g_t > 0 \\
G_{t-1}, & g_t = 0
\end{cases}
\]

\[
L_t =
\begin{cases}
\frac{(t-1)L_{t-1}+\ell_t}{t}, & \ell_t > 0 \\
L_{t-1}, & \ell_t = 0
\end{cases}
\]

预热结束后，相同的方向性更新改用 (n)：

\[
G_t =
\begin{cases}
\frac{(n-1)G_{t-1}+g_t}{n}, & g_t > 0 \\
G_{t-1}, & g_t = 0
\end{cases}
\qquad
L_t =
\begin{cases}
\frac{(n-1)L_{t-1}+\ell_t}{n}, & \ell_t > 0 \\
L_{t-1}, & \ell_t = 0
\end{cases}
\]

输出为：

\[
RS_t = \frac{G_t/n}{L_t/n}, \qquad
RSI_t = 100 - \frac{100}{1 + RS_t}
\]

如果下跌状态为零，越过最初的单样本路径后，RTTA 返回 `100.0`。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RSI` 中实现。上述方向状态的更新方式与当前 C++ 代码完全一致。

## 参考资料

- [ChartSchool：相对强弱指数](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi)
