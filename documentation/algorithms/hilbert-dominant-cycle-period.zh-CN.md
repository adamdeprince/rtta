# Hilbert 主导周期长度（HilbertDominantCyclePeriod）

## 摘要

`HilbertDominantCyclePeriod` 是 RTTA 对市场主导周期长度的流式估计，采用 John Ehlers 的 Hilbert 变换自适应方法，并移植自 TA-Lib `HT_DCPERIOD`。输出是经平滑的周期长度（单位为 K 线根数），通常限制在 \([6,50]\) 范围内。

## 更新 API

```python
result = rtta.HilbertDominantCyclePeriod(fillna=True).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `fillna`  | `True`  | 若为 `False`，超过 TA-Lib 的 32 样本回看期之前返回 NaN |

`update(value)` 处理一个标量价格，并返回当前平滑后的主导周期长度。该指标没有可调周期参数；引擎完全自适应。

## 工作原理

所有 Hilbert 系列指标共用 `src/rtta/indicator.cpp` 中的 `HilbertCycleEngine`。每根 K 线依次执行：

1. 对价格作 **4 根 K 线 WMA 平滑**：\((4x_t+3x_{t-1}+2x_{t-2}+x_{t-3})/10\)。
2. 对平滑值应用**奇偶分离的 Hilbert 去趋势器**：使用 FIR 风格的固定 Hilbert 抽头 \(a=0.0962\)、\(b=0.5769\)，并乘以自适应因子 \(0.075\cdot\text{period}+0.54\)。
3. **同相 / 正交**分量在奇偶 K 线上交替推进；进一步的 Hilbert 级产生 \(I_2,Q_2\)。
4. **复数自相关**更新实部与虚部 \(Re,Im\)；瞬时周期通过 \(Im/Re\) 的 `atan2` 风格角度求得，即 \(360^\circ/\text{angle}\)，随后限制变化速率、硬性截断到 \([6,50]\)，再作 EMA 平滑。

`HilbertDominantCyclePeriod` 只公开最终的**平滑周期** `smooth_period_`。

## 递推公式

### 价格平滑

\[
S_t = \frac{4 x_t + 3 x_{t-1} + 2 x_{t-2} + x_{t-3}}{10}
\]

### 自适应 Hilbert 缩放

\[
\lambda_t = 0.075\, P_{t-1} + 0.54
\]

去趋势 / 正交级应用 TA-Lib 的奇偶 Hilbert 算子 \(H_a(\cdot)\)（系数 \(a=0.0962\)、\(b=0.5769\)，三个槽位的环形缓冲区），并乘以 \(\lambda_t\)。

### 零差鉴频器

使用平滑后的复分量 \(I2_t,Q2_t\)：

\[
Re_t = 0.8\, Re_{t-1} + 0.2\,(I2_t I2_{t-1} + Q2_t Q2_{t-1})
\]

\[
Im_t = 0.8\, Im_{t-1} + 0.2\,(I2_t Q2_{t-1} - Q2_t I2_{t-1})
\]

\[
\tilde{P}_t =
\begin{cases}
\dfrac{360}{\operatorname{atan}(Im_t/Re_t)\cdot\frac{180}{\pi}} & Re_t,Im_t \ne 0 \\
P_{t-1} & \text{其他情况}
\end{cases}
\]

（实现使用 `atan(im/re) * rad2deg`，其中 `rad2deg = 45/atan(1)`。）

### 截断与平滑

\[
\tilde{P}_t \leftarrow \operatorname{clip}\!\big(\tilde{P}_t,\; 0.67 P_{t-1},\; 1.5 P_{t-1}\big)
\]

\[
\tilde{P}_t \leftarrow \operatorname{clip}(\tilde{P}_t,\; 6,\; 50)
\]

\[
P_t = 0.2\,\tilde{P}_t + 0.8\, P_{t-1}
\]

\[
\overline{P}_t = 0.33\, P_t + 0.67\, \overline{P}_{t-1}
\]

返回值为 \(\overline P_t\)（`smooth_period_`）。

当 `fillna=False` 时，在完成超过 32 次更新（TA-Lib `HT_DCPERIOD` 回看期）之前，输出为 NaN。

## 实现说明

- 对 `src/rtta/indicator.cpp` 中 `HilbertCycleEngine::period()` 的轻量封装（`class HilbertDominantCyclePeriod`）。
- 共用状态也驱动相位、相量、正弦波、趋势线和趋势模式。
- 回看期常数：`lookback_period_ = 32`。

## 参考资料

- [TA-Lib HT_DCPERIOD](https://ta-lib.org/functions/ht_dcperiod)
- [Ehlers——Optimal Adaptive Averages（MESA）](https://www.mesasoftware.com/papers/OptimalAdaptiveAverage.pdf)
