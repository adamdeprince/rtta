# Hilbert 主导周期相位（HilbertDominantCyclePhase）

## 摘要

`HilbertDominantCyclePhase` 是 RTTA 对主导周期相位（角度制）的流式实现，移植自 TA-Lib `HT_DCPHASE`。它对当前主导周期内经 Hilbert 平滑的价格作短窗口 DFT 风格求和，再配合 TA-Lib 的滞后与象限修正恢复相位。

## 更新 API

```python
result = rtta.HilbertDominantCyclePhase(fillna=True).update(value)
```

| 参数 | 默认值 | 含义 |
|-----------|---------|---------|
| `fillna`  | `True`  | 若为 `False`，超过 63 个样本的回看期之前返回 NaN |

`update(value)` 返回当前主导周期相位，单位为度。

## 工作原理

`HilbertCycleEngine` 估计出平滑主导周期 \(\overline P_t\) 后，把最近 \(\lfloor\overline P_t+0.5\rfloor\) 个平滑价格（限制为 1…50）视作正弦/余弦基的一整个周期。该复数和的辐角就是周期相位。随后 TA-Lib 应用固定偏移（\(+90^\circ\)）、周期滞后 \(360/\overline P\)、余弦和为负时的半平面翻转，以及相位大于 \(315^\circ\) 时的回绕，使相位与其正弦波及趋势模式逻辑对齐。

随着市场完成一次主导振荡，相位会贯穿整个周期向前推进；跳跃或停滞表示市场状态发生变化，或周期估计不可靠。

## 递推公式

令 \(\overline P_t\) 为 [`HilbertDominantCyclePeriod`](hilbert-dominant-cycle-period.zh-CN.md) 给出的平滑周期，并令：

\[
N = \operatorname{clip}\!\big(\lfloor \overline{P}_t + 0.5 \rfloor,\; 1,\; 50\big).
\]

平滑价格环形缓冲区为 \(S^{(0)},\ldots,S^{(49)}\)（索引从当前平滑值位置向后回溯）：

\[
Real = \sum_{i=0}^{N-1} \sin\!\Big(\frac{2\pi i}{N}\Big)\, S^{(-i)},\qquad
Imag = \sum_{i=0}^{N-1} \cos\!\Big(\frac{2\pi i}{N}\Big)\, S^{(-i)}
\]

基础角度（角度制，与 C++ 的 `atan * rad2deg` 一致）：

\[
\phi \leftarrow
\begin{cases}
\operatorname{atan}(Real/Imag)\cdot\frac{180}{\pi} & |Imag| > 0 \\
\phi - 90 & |Imag|\le 0.01 \land Real < 0 \\
\phi + 90 & |Imag|\le 0.01 \land Real > 0
\end{cases}
\]

TA-Lib 修正：

\[
\phi \leftarrow \phi + 90
\]

\[
\phi \leftarrow \phi + \frac{360}{\overline{P}_t}\quad (\overline{P}_t \ne 0)
\]

\[
\phi \leftarrow \phi + 180 \quad \text{若 } Imag < 0
\]

\[
\phi \leftarrow \phi - 360 \quad \text{若 } \phi > 315
\]

返回值：\(\phi_t=\phi\)。

当 `fillna=False` 时，在完成超过 63 次更新之前，输出为 NaN。

## 实现说明

- 对 `HilbertCycleEngine::phase()` 的轻量封装（`class HilbertDominantCyclePhase`）。
- 回看期常数：`lookback_phase_ = 63`。
- 与其他 Hilbert 指标使用相同的引擎实例状态；每个公开类都持有自己的引擎。

## 参考资料

- [TA-Lib HT_DCPHASE](https://ta-lib.org/functions/ht_dcphase)
- [MESA Software——Ehlers Hilbert 论文](https://www.mesasoftware.com/)
