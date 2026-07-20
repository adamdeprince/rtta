# Hilbert 趋势模式（HilbertTrendMode）

## 摘要

`HilbertTrendMode` 是 RTTA 从 Hilbert 引擎得到的趋势/周期市场状态流式标志（TA-Lib `HT_TRENDMODE`）。趋势模式返回 `1`，周期模式返回 `0`。

## 更新 API

```python
result = rtta.HilbertTrendMode(fillna=True).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `fillna`  | `True`  | 若为 `False`，超过 63 个样本的回看期之前返回 NaN |

`update(value)` 返回 `1.0`（趋势）或 `0.0`（周期）；当 `fillna=False` 时，预热期间返回 NaN。

## 工作原理

Ehlers / TA-Lib 使用 Hilbert 引擎已经计算出的信号，把每根 K 线归类为**周期**或**趋势**：

1. **正弦波 / 超前正弦波交叉** → 强制进入周期模式并重置趋势持续根数。
2. **趋势持续时间较短**——若距最近周期事件的 K 线根数少于平滑周期的一半，则保持周期模式。
3. **相位有序推进**——若相位增加量约等于一个周期步长（\(360/\overline P\) 的 \(0.67\) 至 \(1.5\) 倍），则保持周期模式。
4. **价格远离趋势线**——若当前 WMA 平滑价格相对 Hilbert 趋势线的偏离至少为 \(1.5\%\)，则强制进入趋势模式。

应用覆盖规则前的默认值为趋势（`1`）。结果是供策略过滤使用的二元市场状态标签，而不是连续的趋势强度指标。

## 递推公式

令 \(sine_t\)、\(lead_t\) 为 Hilbert 正弦波对，\(\phi_t\) 为主导周期相位，\(\overline P_t\) 为平滑周期，\(S_t\) 为 4 根 K 线 WMA 价格，\(TL_t\) 为 Hilbert 趋势线。维护整数 `days_in_trend`。

每根 K 线开始时令 \(trend\leftarrow1\)。

**发生交叉 → 周期模式并重置持续根数：**

\[
\begin{aligned}
&\text{若 } (sine_t > lead_t \land sine_{t-1} \le lead_{t-1}) \\
&\quad\text{或 } (sine_t < lead_t \land sine_{t-1} \ge lead_{t-1}): \\
&\qquad days \leftarrow 0,\; trend \leftarrow 0
\end{aligned}
\]

随后 \(days\leftarrow days+1\)。

**趋势持续时间较短 → 周期模式：**

\[
\text{若 } days < 0.5\,\overline{P}_t:\quad trend \leftarrow 0
\]

**相位步长位于区间内 → 周期模式：**

\[
\Delta\phi = \phi_t - \phi_{t-1}
\]

\[
\text{若 } \overline{P}_t \ne 0 \text{ 且 }
0.67\cdot\frac{360}{\overline{P}_t} < \Delta\phi < 1.5\cdot\frac{360}{\overline{P}_t}:
\quad trend \leftarrow 0
\]

**远离趋势线 → 趋势模式：**

\[
\text{若 } TL_t \ne 0 \text{ 且 }
\left|\frac{S_t - TL_t}{TL_t}\right| \ge 0.015:
\quad trend \leftarrow 1
\]

返回值：\(trendmode_t=trend\)，类型为 double（`0.0` 或 `1.0`）。

当 `fillna=False` 时，在完成超过 63 次更新之前返回 NaN。

## 实现说明

- 对 `HilbertCycleEngine::trendmode()` 的轻量封装（`class HilbertTrendMode`）。
- 依赖引擎同一次 `update` 内部的正弦波、相位、平滑价格和趋势线路径。
- 回看期：`lookback_phase_ = 63`。

## 参考资料

- [TA-Lib HT_TRENDMODE](https://ta-lib.org/functions/ht_trendmode)
- [MESA Software——Ehlers 论文](https://www.mesasoftware.com/)
