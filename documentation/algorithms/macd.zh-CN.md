# MACD

## 摘要

`MACD` 是 RTTA 的标量移动平均收敛/发散计算管线。它分别计算快速 EMA 和慢速 EMA，求两者之差，再返回该差值的 EMA 信号线。

## 更新 API

```python
value = rtta.MACD(a=12, b=26, c=9, fillna=False).update(value)
```

`a` 是快速 EMA 的窗口，`b` 是慢速 EMA 的窗口，`c` 是信号 EMA 的窗口。

## 工作原理

MACD 衡量快速趋势估计与慢速趋势估计之间的距离。RTTA 的标量 `MACD` 类依次计算：

- 输入值的快速 EMA；
- 输入值的慢速 EMA；
- 两者的原始差值；
- 原始差值的信号 EMA。

类的返回值是最后一项，即经过信号平滑的线。若需要带百分比振荡值、信号线和柱状图字段的多字段结果，请参阅 `PercentagePrice`。

## 递推公式

令 \(x_t\) 为输入，并定义平滑系数为 \(\alpha_a=2/(1+a)\)、\(\alpha_b=2/(1+b)\) 和 \(\alpha_c=2/(1+c)\) 的 EMA 递推式。

\[
F_t = \alpha_a x_t + (1-\alpha_a)F_{t-1}
\]

\[
S_t = \alpha_b x_t + (1-\alpha_b)S_{t-1}
\]

\[
D_t = F_t - S_t
\]

\[
M_t = \alpha_c D_t + (1-\alpha_c)M_{t-1}
\]

`update(...)` 返回 \(M_t\)。当 `fillna=False` 时，在内部计数器达到 \(\max(a,b)+c\) 之前返回 `NaN`，但所有 EMA 状态仍会正常推进。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MACD` 中实现。

## 参考资料

- [ChartSchool：MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
