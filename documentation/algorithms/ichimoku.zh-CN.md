# Ichimoku

## 摘要

`Ichimoku` 是一组流式一目均衡表分量：转换线（Tenkan-sen）、基准线（Kijun-sen）、先行带 A 与先行带 B（未经延迟的因果值）、迟行带（Chikou），以及**延迟 `window2` 根 K 线的云带**。后两条延迟云带与传统图表绘制云层的 K 线位置对齐（`span_a_displaced`、`span_b_displaced`）。

## 更新 API

```python
import rtta

ind = rtta.Ichimoku(window1=9, window2=26, window3=52, fillna=True)
result = ind.update(high, low, close)
# result.conversion, result.base, result.span_a, result.span_b,
# result.lagging_span,
# result.span_a_displaced, result.span_b_displaced
```

`advance(...)` 更新状态但不返回结果。批量接口为全部七个字段返回并行数组。

## 工作原理

一目均衡表使用滚动最高价—最低价区间的中点：

| 分量 | 经典名称 | 定义 |
|-----------|--------------|------------|
| conversion | Tenkan-sen（转换线） | `window1` 内最高价与最低价的中点 |
| base | Kijun-sen（基准线） | `window2` 内最高价与最低价的中点 |
| span_a | Senkou Span A（原始先行带 A） | 转换线与基准线的中点 |
| span_b | Senkou Span B（原始先行带 B） | `window3` 内最高价与最低价的中点 |
| lagging_span | Chikou Span（迟行带） | 收盘价延迟 `window2` 根 K 线 |
| span_a_displaced | 当前 K 线处的云带 A | 原始 span_a 延迟 `window2` 根 K 线 |
| span_b_displaced | 当前 K 线处的云带 B | 原始 span_b 延迟 `window2` 根 K 线 |

在经典图表中，先行带会向**未来**绘制 26 根 K 线。因果数据流无法输出未来值，因此 RTTA 也会输出在当前 K 线处**到达**的先行带：即 `window2` 个时点前计算的数值。未经延迟的 `span_a` / `span_b` 仍然可用于自定义先行云逻辑。

当 `fillna=False` 时，区间窗口未填满的分量返回 NaN。延迟线使用相同的 `fillna` 策略；填充值模式下，原始先行带为 NaN 时以 `close` 替换，使延迟缓冲区始终有定义。

## 递推公式

令 \(n_1,n_2,n_3\) 分别为 `window1`、`window2`、`window3`。在最近 (n) 个最高价/最低价上计算滚动极值：

\[
\begin{aligned}
C_t &= \tfrac12\bigl(\max_{i\in[t-n_1+1,t]} H_i + \min_{i\in[t-n_1+1,t]} L_i\bigr),\\
B_t &= \tfrac12\bigl(\max_{i\in[t-n_2+1,t]} H_i + \min_{i\in[t-n_2+1,t]} L_i\bigr),\\
A_t &= \tfrac12(C_t + B_t),\\
S_t &= \tfrac12\bigl(\max_{i\in[t-n_3+1,t]} H_i + \min_{i\in[t-n_3+1,t]} L_i\bigr).
\end{aligned}
\]

迟行带与位移云带（延迟长度为 \(n_2\)）：

\[
\mathrm{lagging\_span}_t = c_{t-n_2}
\quad\text{（收盘价延迟；若 `fillna=False`，延迟填满前为 NaN）},
\]

\[
\begin{aligned}
A^{\mathrm{in}}_t &=
\begin{cases}
A_t, & A_t \text{ 为有限值},\\
c_t, & \text{为 NaN 且启用 `fillna`},\\
\text{NaN}, & \text{为 NaN 且未启用 `fillna`},
\end{cases}
\qquad
\mathrm{span\_a\_displaced}_t = A^{\mathrm{in}}_{t-n_2},
\end{aligned}
\]

\(S_t\rightarrow\mathrm{span\_b\_displaced}_t\) 的计算方式相同。

公开字段：`conversion`\(=C_t\)、`base`\(=B_t\)、`span_a`\(=A_t\)、`span_b`\(=S_t\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class Ichimoku` 中实现，使用 `RollingExtreme` 计算最高价/最低价，并使用 `Delay` 计算迟行带和位移先行带。结果类型为 `IchimokuResult` / `IchimokuBatchResult`。

## 参考资料

- [Investopedia——Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp)
