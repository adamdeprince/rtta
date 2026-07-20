# Hilbert 相量（HilbertPhasor）

## 摘要

`HilbertPhasor` 是 RTTA 对 Hilbert 同相分量与正交分量的流式实现，移植自 TA-Lib `HT_PHASOR`。二者共同构成经过去趋势和平滑处理的价格复解析信号，供主导周期估计器内部使用。

## 更新 API

```python
result = rtta.HilbertPhasor(fillna=True).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `fillna`  | `True`  | 若为 `False`，超过 32 个样本的回看期之前返回 NaN |

`update(...)` 返回的结果包含：

- `inphase`——同相分量 \(I\)
- `quadrature`——正交分量 \(Q\)

`advance(value)` 更新状态；`last()` 返回缓存的结果。

## 工作原理

Hilbert 变换会为实信号生成一个相移 \(90^\circ\) 的伴随信号。Ehlers / TA-Lib 先对价格的 4 根 K 线 WMA 应用短 FIR 风格的 Hilbert 算子（即去趋势器），再取：

- **同相分量**：去趋势值经过三个自适应 Hilbert 级后的延迟值（根据 K 线奇偶性使用 `i1_for_even_prev3_` 或 `i1_for_odd_prev3_`）。
- **正交分量**：去趋势值的 Hilbert 变换（\(Q1\)）。

复数对 \((I,Q)\) 被送入估计主导周期的零差鉴频器。绘制 \(I\) 对 \(Q\) 的图形，可以得到局部周期的相量图。

## 递推公式

前端与其他 Hilbert 指标共用（参见 [`HilbertDominantCyclePeriod`](hilbert-dominant-cycle-period.zh-CN.md)）：

\[
S_t = \frac{4 x_t + 3 x_{t-1} + 2 x_{t-2} + x_{t-3}}{10}
\]

\[
\lambda_t = 0.075\, P_{t-1} + 0.54
\]

令 \(D_t\) 为 \(S_t\) 经自适应 Hilbert 去趋势器处理后的结果，\(Q1_t\) 为 \(D_t\) 的自适应 Hilbert 变换。K 线奇偶性决定交替使用的滤波器组：

\[
I_t = D_{t-3}^{\text{（奇偶路径）}},\qquad
Q_t = Q1_t
\]

每次更新后，代码中的输出为：

\[
\texttt{inphase} = I_t,\qquad
\texttt{quadrature} = Q_t
\]

当 `fillna=False` 时，在处理超过 32 个样本之前，两个字段均为 NaN。

## 实现说明

- 以 `class HilbertPhasor` 实现，封装 `HilbertCycleEngine::inphase()` 与 `::quadrature()`。
- 结果类型：`HilbertPhasorResult`（`inphase`、`quadrature`）。
- 回看期与 `HT_DCPERIOD` / `HT_PHASOR` 一致（32）。
- 批量辅助函数：`batch_hilbert_phasor`。

## 参考资料

- [TA-Lib HT_PHASOR](https://ta-lib.org/functions/ht_phasor)
- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
