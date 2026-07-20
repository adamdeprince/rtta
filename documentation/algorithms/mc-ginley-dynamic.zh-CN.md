# McGinley 动态平均线（McGinleyDynamic）

## 摘要

`McGinleyDynamic` 是 RTTA 对 McGinley Dynamic 的流式实现：一种自适应移动平均线。当价格远离均线时，它会自动加快；价格靠近时则放慢。与速度固定的 EMA 相比，可以减少来回穿越产生的虚假信号。

## 更新 API

```python
result = rtta.McGinleyDynamic(window=14, fillna=True).update(price)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `window`  | `14`    | 速度常数 \(N\) |
| `fillna`  | `True`  | 若为 `False`，取得 `window` 个样本前返回 NaN |

`update(price)` 以标量形式返回当前 McGinley Dynamic 值。

## 工作原理

John R. McGinley 设计 Dynamic 的目的，是在无需不断重新调整周期的情况下，比简单或指数移动平均更贴近价格。向价格迈进的步长除以 \(N(price/MD)^4\)：

- 当价格远高于 MD 时，比率 \(>1\)，分母增大，但较大的分子仍会向上拉动 MD；四次幂响应经过调校，使均线平滑地“赶上”价格，而不像慢速 SMA 那样滞后。
- 当价格在 MD 附近振荡时，有效平滑程度更高，从而抑制噪声。

首个观测将 \(MD\) 初始化为价格。若 MD 为零，则重新以价格初始化；分母不是有限值时也会重新初始化。

## 递推公式

令 \(z_t\) 为价格，\(N=\max(\texttt{window},1)\)。初始化：

\[
MD_1 = z_1
\]

对于 \(t>1\)，若 \(MD_{t-1}=0\)，令 \(MD_t=z_t\)；否则令 \(r_t=z_t/MD_{t-1}\)（安全除法默认值为 1）：

\[
D_t = N \cdot r_t^{4}
\]

\[
MD_t =
\begin{cases}
MD_{t-1} + \dfrac{z_t - MD_{t-1}}{D_t} & D_t \ne 0 \land D_t \text{ 为有限值} \\
z_t & \text{其他情况}
\end{cases}
\]

当 `fillna=False` 且取得的样本少于 \(N\) 个时，返回 NaN；否则返回 \(MD_t\)。

## 实现说明

- 在 `src/rtta/indicator.cpp` 的 `class McGinleyDynamic` 中实现。
- 比率使用 `safe_divide(price, value_, 1.0)`。
- 输出为标量 `double`。

## 参考资料

- [Investopedia——McGinley Dynamic](https://www.investopedia.com/terms/m/mcginley-dynamic.asp)
- [McGinley Dynamic 概述](https://www.tradingview.com/support/solutions/43000589132-mcginley-dynamic/)
